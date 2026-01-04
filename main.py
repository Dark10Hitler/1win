import asyncio
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
import logging
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LuckyJet High-Speed Predictor")

# ============= МАТЕМАТИЧЕСКОЕ ЯДРО (ОПТИМИЗИРОВАНО) =============
import hashlib
import json
import requests

# Ваши данные из панели Cryptomus
MERCHANT_ID = "ВАШ_ID"
API_KEY = "ВАШ_API_KEY"

@app.post("/create-payment")
async def create_payment(user_id: str):
    payload = {
        "amount": "30.00",
        "currency": "USD",
        "order_id": f"sub_{user_id}_{int(datetime.now().timestamp())}",
        "url_callback": "https://ваш-бэкенд.render.com/webhook/cryptomus"
    }
    
    # Генерация подписи (Signature) по документации Cryptomus
    data_json = json.dumps(payload)
    sign = hashlib.md5(
        (base64.b64encode(data_json.encode()).decode() + API_KEY).encode()
    ).hexdigest()
    
    headers = {
        "merchant": MERCHANT_ID,
        "sign": sign,
        "Content-Type": "application/json"
    }
    
    response = requests.post("https://api.cryptomus.com/v1/payment", headers=headers, data=data_json)
    return response.json() # Возвращает ссылку на оплату в поле 'url'
    
class FastModelManager:
    def __init__(self):
        self.model = None
        self.features_names = None
        self.is_trained = False
        self._lock = asyncio.Lock()

    def build_features_row(self, mults: List[float], M: float, lags=10) -> np.ndarray:
        """Быстрое извлечение признаков без использования Pandas (для скорости)"""
        n = len(mults)
        # Lag features
        feats = []
        for i in range(1, lags + 1):
            feats.append(mults[-i] if n >= i else 1.0)
        
        # Window stats
        window = mults[-20:] if n >= 20 else mults
        feats.append(np.mean(window))
        feats.append(np.std(window))
        feats.append(np.max(window))
        
        # Probability features
        feats.append(sum(1 for x in mults if x >= M) / max(1, n))
        
        # Recent run length (low)
        run = 0
        for x in reversed(mults):
            if x < 1.5: run += 1
            else: break
        feats.append(float(run))
        
        # Since last ge M
        since = 0
        for i, x in enumerate(reversed(mults), start=1):
            if x >= M:
                since = i
                break
        feats.append(float(since if since > 0 else n + 1))
        
        return np.array(feats).reshape(1, -1)

    async def train_model(self, all_mults: List[float], M_train: float = 2.0):
        """Фоновое обучение (вызывать вне основного потока прогноза)"""
        async with self._lock:
            n = len(all_mults)
            if n < 50: return
            
            X, y = [], []
            # Собираем обучающую выборку
            for i in range(20, n - 1):
                X.append(self.build_features_row(all_mults[:i+1], M_train).flatten())
                y.append(1 if all_mults[i+1] >= M_train else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Быстрая логистическая регрессия с калибровкой
            clf = LogisticRegression(solver='liblinear')
            calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
            calibrated.fit(X, y)
            
            self.model = calibrated
            self.is_trained = True
            logger.info(f"Model retrained on {len(y)} samples")

    def quick_predict(self, recent_mults: List[float], M: float) -> float:
        """Мгновенный точечный прогноз (миллисекунды)"""
        if not self.is_trained:
            # Если не обучено, используем эмпирическую вероятность
            return sum(1 for x in recent_mults if x >= M) / max(1, len(recent_mults))
        
        feats = self.build_features_row(recent_mults, M)
        return float(self.model.predict_proba(feats)[0, 1])

model_mgr = FastModelManager()

# ============= УТИЛИТЫ =============

def calculate_kelly(p: float, M: float, bankroll: float, risk_cap: float) -> dict:
    """Математический расчет по критерию Келли"""
    b = M - 1.0
    if b <= 0: return {"fraction": 0, "amount": 0}
    
    q = 1.0 - p
    f = (b * p - q) / b
    
    # Консервативный Келли (f/4)
    suggested_f = max(0, f / 4.0)
    final_f = min(suggested_f, risk_cap)
    
    return {
        "fraction": round(final_f, 4),
        "amount": round(final_f * bankroll, 2),
        "raw_kelly": round(f, 4)
    }

# ============= API ЭНДПОИНТЫ =============

class PredictRequest(BaseModel):
    history: List[float]
    target_m: float = 2.0
    bankroll: float = 1000.0
    risk_cap: float = 0.05

@app.post("/predict")
async def predict(req: PredictRequest):
    start_time = datetime.now()
    
    # 1. Получаем вероятность (очень быстро)
    p_win = model_mgr.quick_predict(req.history, req.target_m)
    
    # 2. Считаем доверительный интервал (упрощенный быстрый метод вместо bootstrap)
    # Используем стандартную ошибку доли: SE = sqrt( p*(1-p) / n )
    n = len(req.history)
    se = np.sqrt((p_win * (1 - p_win)) / n)
    ci_lo = max(0, p_win - 1.96 * se)
    ci_hi = min(1, p_win + 1.96 * se)
    
    # 3. Расчет ставки
    kelly = calculate_kelly(p_win, req.target_m, req.bankroll, req.risk_cap)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "status": "success",
        "prediction": {
            "p_win": round(p_win, 4),
            "ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "recommendation": kelly,
            "decision": "BET" if kelly["amount"] > 0 and p_win > (1/req.target_m) else "WAIT"
        },
        "meta": {
            "execution_time_sec": execution_time,
            "history_size": n
        }
    }

@app.post("/train")
async def train(history: List[float]):
    """Эндпоинт для фонового обучения"""
    asyncio.create_task(model_mgr.train_model(history))
    return {"message": "Training started in background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
