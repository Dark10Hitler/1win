import asyncio
import logging
import hashlib
import json
import base64
import numpy as np
import requests
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aiogram import Bot, Dispatcher, types, Router
from aiogram.filters import CommandStart
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# --- НАСТРОЙКИ ---
BOT_TOKEN = "7892346072:AAGrdFAP5WRlctKWgmnO8egTHuoJs0kKZXc"
MERCHANT_ID = "ВАШ_ID"  # Замени на свой из Cryptomus
API_KEY = "ВАШ_API_KEY"   # Замени на свой из Cryptomus
FRONTEND_URL = "https://your-lovable-site.vercel.app" # URL твоего сайта

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LuckyJet Quantum Predictor")

# Разрешаем запросы с фронтенда Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= МАТЕМАТИЧЕСКОЕ ЯДРО (ML) =============

class FastModelManager:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self._lock = asyncio.Lock()

    def build_features_row(self, mults: List[float], M: float, lags=10) -> np.ndarray:
        n = len(mults)
        feats = []
        # Лаги (прошлые значения)
        for i in range(1, lags + 1):
            feats.append(mults[-i] if n >= i else 1.0)
        
        # Статистика окна
        window = mults[-20:] if n >= 20 else mults
        feats.extend([np.mean(window), np.std(window), np.max(window)])
        
        # Вероятностные признаки
        feats.append(sum(1 for x in mults if x >= M) / max(1, n))
        
        # Длина текущей серии "низких" коэфф
        run = 0
        for x in reversed(mults):
            if x < 1.5: run += 1
            else: break
        feats.append(float(run))
        
        # Сколько раундов назад был коэфф >= M
        since = 0
        for i, x in enumerate(reversed(mults), start=1):
            if x >= M:
                since = i
                break
        feats.append(float(since if since > 0 else n + 1))
        
        return np.array(feats).reshape(1, -1)

    async def train_model(self, all_mults: List[float], M_train: float = 2.0):
        async with self._lock:
            n = len(all_mults)
            if n < 50: return
            
            X, y = [], []
            for i in range(20, n - 1):
                X.append(self.build_features_row(all_mults[:i+1], M_train).flatten())
                y.append(1 if all_mults[i+1] >= M_train else 0)
            
            clf = LogisticRegression(solver='liblinear')
            calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
            calibrated.fit(np.array(X), np.array(y))
            
            self.model = calibrated
            self.is_trained = True
            logger.info(f"Model trained on {len(y)} rounds")

    def quick_predict(self, recent_mults: List[float], M: float) -> float:
        if not self.is_trained:
            return sum(1 for x in recent_mults if x >= M) / max(1, len(recent_mults))
        
        feats = self.build_features_row(recent_mults, M)
        return float(self.model.predict_proba(feats)[0, 1])

model_mgr = FastModelManager()

# ============= КРИПТО ОПЛАТА (CRYPTOMUS) =============

@app.post("/create-payment")
async def create_payment(user_id: str):
    payload = {
        "amount": "30.00",
        "currency": "USD",
        "order_id": f"sub_{user_id}_{int(datetime.now().timestamp())}",
        "url_callback": f"{FRONTEND_URL}/webhook/cryptomus"
    }
    
    data_json = json.dumps(payload)
    # Правильная генерация подписи для Cryptomus
    sign = hashlib.md5(
        (base64.b64encode(data_json.encode()).decode() + API_KEY).encode()
    ).hexdigest()
    
    headers = {"merchant": MERCHANT_ID, "sign": sign, "Content-Type": "application/json"}
    
    try:
        response = requests.post("https://api.cryptomus.com/v1/payment", headers=headers, data=data_json)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= API ПРОГНОЗИРОВАНИЯ =============

class PredictRequest(BaseModel):
    history: List[float]
    target_m: float = 2.0
    bankroll: float = 1000.0
    risk_cap: float = 0.05

@app.post("/predict")
async def predict(req: PredictRequest):
    start_time = datetime.now()
    p_win = model_mgr.quick_predict(req.history, req.target_m)
    
    # Расчет по Келли
    b = req.target_m - 1.0
    q = 1.0 - p_win
    f = (b * p_win - q) / b if b > 0 else 0
    suggested_f = min(max(0, f / 4.0), req.risk_cap) # f/4 для безопасности
    
    # Доверительный интервал
    se = np.sqrt((p_win * (1 - p_win)) / max(1, len(req.history)))
    
    return {
        "status": "success",
        "prediction": {
            "p_win": round(p_win, 4),
            "ci": [round(max(0, p_win - 1.96*se), 4), round(min(1, p_win + 1.96*se), 4)],
            "bet_amount": round(suggested_f * req.bankroll, 2),
            "decision": "BET" if suggested_f > 0 and p_win > (1/req.target_m) else "WAIT"
        },
        "meta": {"exec_time": (datetime.now() - start_time).total_seconds()}
    }

@app.post("/train")
async def train_endpoint(history: List[float], background_tasks: BackgroundTasks):
    background_tasks.add_task(model_mgr.train_model, history)
    return {"message": "Training started"}

# ============= ТЕЛЕГРАМ БОТ =============

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
router = Router()

@router.message(CommandStart())
async def cmd_start(message: types.Message):
    kb = [[types.InlineKeyboardButton(text="Запустить Терминал 🚀", url=FRONTEND_URL)]]
    markup = types.InlineKeyboardMarkup(inline_keyboard=kb)
    await message.answer(
        "💎 **Quantum Predictor v2.1** запущен.\n\n"
        "Для работы используйте веб-интерфейс нажав на кнопку ниже.",
        reply_markup=markup
    )

dp.include_router(router)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(dp.start_polling(bot))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
