# -*- coding: utf-8 -*-
"""
Backtesting y trading dual-timeframe optimizado con almacenamiento SQLite
Timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h
Apalancamiento dinámico basado en win_rate
Ejecutable en Railway o cualquier backend
"""

import requests
import pandas as pd
import numpy as np
import sqlite3
import time
import json
import os
from datetime import datetime, timedelta
from itertools import product

# ==================== CONFIGURACIÓN ====================
SYMBOL = 'BTCUSDT'
INTERVAL_BASE = '1m'
HOURS = 48                      # Datos históricos para optimización
LIMIT = 1000
DB_PATH = 'trading_bot.db'      # Base de datos SQLite

# Parámetros de simulación realista
SLIPPAGE = 0.001                # 0.1% deslizamiento
COMMISSION = 0.001               # 0.1% comisión por operación
BASE_CAPITAL = 1000              # Capital base en USD
MAX_LEVERAGE = 20                 # Apalancamiento máximo permitido
MIN_WIN_RATE_FOR_LEVERAGE = 0.4  # Mínimo win rate para usar apalancamiento

# Rangos de optimización
ADX_RANGE = [20, 25, 30, 35]
RSI_LOW_RANGE = [20, 25, 30, 35]
RSI_HIGH_RANGE = [65, 70, 75, 80]
MULT_STOP_RANGE = [1.0, 1.5, 2.0, 2.5]
MULT_TP_RANGE = [1.0, 1.5, 2.0, 2.5]

# Timeframes disponibles
TIMEFRAMES = {
    '1m': '1min',
    '3m': '3min',
    '5m': '5min',
    '15m': '15min',
    '30m': '30min',
    '1h': '1h',
    '2h': '2h',
    '4h': '4h'
}

# Combinaciones de timeframes a probar (entrada, tendencia)
TF_COMBINATIONS = [
    ('1m', '5m'),
    ('1m', '15m'),
    ('1m', '1h'),
    ('3m', '15m'),
    ('3m', '1h'),
    ('5m', '1h'),
    ('5m', '4h'),
    ('15m', '1h'),
    ('15m', '4h'),
    ('30m', '4h')
]

# ==================== BASE DE DATOS ====================
def init_database():
    """Crea las tablas necesarias si no existen."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Tabla de resultados de optimización
    c.execute('''
        CREATE TABLE IF NOT EXISTS optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            tf_entrada TEXT,
            tf_tendencia TEXT,
            adx_th INTEGER,
            rsi_low INTEGER,
            rsi_high INTEGER,
            mult_stop REAL,
            mult_tp REAL,
            use_slope BOOLEAN,
            profit REAL,
            trades INTEGER,
            win_rate REAL,
            max_dd REAL
        )
    ''')
    
    # Tabla de trades individuales (para backtest y live)
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            combo_id INTEGER,
            entry_time DATETIME,
            exit_time DATETIME,
            tipo TEXT,
            entry_price REAL,
            exit_price REAL,
            retorno REAL,
            razon TEXT,
            win BOOLEAN,
            leverage_used REAL,
            capital_used REAL,
            FOREIGN KEY(combo_id) REFERENCES optimization_results(id)
        )
    ''')
    
    # Tabla de parámetros en uso (para trading live)
    c.execute('''
        CREATE TABLE IF NOT EXISTS active_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tf_entrada TEXT,
            tf_tendencia TEXT,
            adx_th INTEGER,
            rsi_low INTEGER,
            rsi_high INTEGER,
            mult_stop REAL,
            mult_tp REAL,
            use_slope BOOLEAN,
            win_rate REAL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_optimization_result(tf_entrada, tf_tendencia, params, metrics):
    """Guarda un resultado de optimización en la BD."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO optimization_results
        (tf_entrada, tf_tendencia, adx_th, rsi_low, rsi_high, mult_stop, mult_tp, use_slope,
         profit, trades, win_rate, max_dd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        tf_entrada, tf_tendencia,
        params['adx_th'], params['rsi_low'], params['rsi_high'],
        params['mult_stop'], params['mult_tp'], int(params['use_slope']),
        metrics['profit'], metrics['trades'], metrics['win_rate'], metrics['max_dd']
    ))
    conn.commit()
    conn.close()

def save_trade(combo_id, trade, entry_time, exit_time, leverage, capital_used):
    """Guarda un trade individual."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO trades
        (combo_id, entry_time, exit_time, tipo, entry_price, exit_price, retorno, razon, win, leverage_used, capital_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        combo_id,
        entry_time.isoformat() if isinstance(entry_time, datetime) else entry_time,
        exit_time.isoformat() if isinstance(exit_time, datetime) else exit_time,
        trade['tipo'],
        trade['entrada'],
        trade['salida'],
        trade['retorno'],
        trade['razon'],
        1 if trade['retorno'] > 0 else 0,
        leverage,
        capital_used
    ))
    conn.commit()
    conn.close()

def get_best_params(tf_entrada, tf_tendencia):
    """Obtiene los mejores parámetros para una combinación desde la BD."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT adx_th, rsi_low, rsi_high, mult_stop, mult_tp, use_slope, win_rate
        FROM optimization_results
        WHERE tf_entrada = ? AND tf_tendencia = ?
        ORDER BY profit DESC
        LIMIT 1
    ''', (tf_entrada, tf_tendencia))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            'adx_th': row[0],
            'rsi_low': row[1],
            'rsi_high': row[2],
            'mult_stop': row[3],
            'mult_tp': row[4],
            'use_slope': bool(row[5]),
            'win_rate': row[6]
        }
    return None

# ==================== INDICADORES ====================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    up_move = high - high.shift()
    down_move = low.shift() - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()
    adx_slope = adx - adx.shift(3)
    df_out = pd.DataFrame(index=df.index)
    df_out['ADX'] = adx
    df_out['DI_plus'] = plus_di
    df_out['DI_minus'] = minus_di
    df_out['ADX_slope'] = adx_slope
    return df_out

def compute_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# ==================== DESCARGA DE DATOS ====================
def fetch_klines(symbol, interval, hours):
    base_url = 'https://api.binance.com/api/v3/klines'
    end_time = int(time.time() * 1000)
    start_time = end_time - hours * 60 * 60 * 1000
    all_klines = []
    current_start = start_time
    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': LIMIT
        }
        try:
            resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_klines.extend(data)
            current_start = data[-1][0] + 1
        except Exception as e:
            print(f"Error en descarga: {e}")
            break
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    df = pd.DataFrame(all_klines, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]

def resample_ohlc(df, rule):
    return df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

# ==================== BACKTEST PRINCIPAL ====================
def backtest_dual_mtf(df_entrada, df_tendencia, params, capital=BASE_CAPITAL, leverage=1):
    """
    Backtest usando dos timeframes.
    Retorna: trades, metrics, trades_details (con timestamps)
    """
    # Alinear índices
    df_tendencia_reindex = df_tendencia.reindex(df_entrada.index, method='ffill')
    
    # Calcular indicadores en entrada
    df_entrada = df_entrada.copy()
    df_entrada['RSI'] = compute_rsi(df_entrada['close'], 14)
    df_entrada['ATR'] = compute_atr(df_entrada, 14)
    
    # Indicadores de tendencia
    adx_df = compute_adx(df_tendencia, 14)
    df_tendencia_reindex = df_tendencia_reindex.join(adx_df)
    
    df = df_entrada.join(df_tendencia_reindex[['ADX', 'DI_plus', 'DI_minus', 'ADX_slope']])
    
    # Variables de estado
    position = None
    entry_price = 0.0
    entry_atr = 0.0
    extreme_price = 0.0
    stop_price = 0.0
    take_profit = 0.0
    entry_time = None
    trades = []
    equity_curve = [0.0]
    
    for idx, row in df.iterrows():
        # Determinar tendencia
        tendencia_long = tendencia_short = False
        if pd.notna(row['ADX']) and pd.notna(row['DI_plus']) and pd.notna(row['DI_minus']):
            tendencia_long = (row['ADX'] > params['adx_th']) and (row['DI_plus'] > row['DI_minus'])
            tendencia_short = (row['ADX'] > params['adx_th']) and (row['DI_minus'] > row['DI_plus'])
            if params.get('use_slope', False):
                tendencia_long = tendencia_long and (row['ADX_slope'] > 0)
                tendencia_short = tendencia_short and (row['ADX_slope'] < 0)
        
        # Señal de entrada por RSI
        entrada_long = entrada_short = False
        if pd.notna(row['RSI']):
            entrada_long = (row['RSI'] < params['rsi_low'])
            entrada_short = (row['RSI'] > params['rsi_high'])
        
        # Gestión de posición
        if position is None:
            if tendencia_long and entrada_long:
                position = 'long'
                entry_price = row['close'] * (1 + SLIPPAGE)
                entry_atr = row['ATR']
                extreme_price = row['high']
                stop_price = extreme_price - params['mult_stop'] * entry_atr
                take_profit = entry_price + params['mult_tp'] * entry_atr
                entry_time = idx
            elif tendencia_short and entrada_short:
                position = 'short'
                entry_price = row['close'] * (1 - SLIPPAGE)
                entry_atr = row['ATR']
                extreme_price = row['low']
                stop_price = extreme_price + params['mult_stop'] * entry_atr
                take_profit = entry_price - params['mult_tp'] * entry_atr
                entry_time = idx
        else:
            exit_reason = None
            exit_price = None
            if position == 'long':
                if row['high'] > extreme_price:
                    extreme_price = row['high']
                    stop_price = extreme_price - params['mult_stop'] * entry_atr
                if row['low'] <= stop_price:
                    exit_reason = 'trailing_stop'
                    exit_price = stop_price
                elif row['high'] >= take_profit:
                    exit_reason = 'take_profit'
                    exit_price = take_profit
                elif tendencia_short:
                    exit_reason = 'tendencia_opuesta'
                    exit_price = row['close']
            else:  # short
                if row['low'] < extreme_price:
                    extreme_price = row['low']
                    stop_price = extreme_price + params['mult_stop'] * entry_atr
                if row['high'] >= stop_price:
                    exit_reason = 'trailing_stop'
                    exit_price = stop_price
                elif row['low'] <= take_profit:
                    exit_reason = 'take_profit'
                    exit_price = take_profit
                elif tendencia_long:
                    exit_reason = 'tendencia_opuesta'
                    exit_price = row['close']
            
            if exit_reason:
                # Aplicar slippage a la salida
                if position == 'long':
                    exit_price_adj = exit_price * (1 - SLIPPAGE)
                    ret = (exit_price_adj - entry_price) / entry_price - COMMISSION
                else:
                    exit_price_adj = exit_price * (1 + SLIPPAGE)
                    ret = (entry_price - exit_price_adj) / entry_price - COMMISSION
                trade = {
                    'tipo': position,
                    'entrada': entry_price,
                    'salida': exit_price_adj,
                    'retorno': ret,
                    'razon': exit_reason
                }
                trades.append(trade)
                equity_curve.append(equity_curve[-1] + ret)
                # Guardar en BD si se pasa combo_id
                if 'combo_id' in params:
                    save_trade(params['combo_id'], trade, entry_time, idx, leverage, capital * leverage)
                position = None
    
    # Cerrar posición al final
    if position is not None:
        last_row = df.iloc[-1]
        exit_price = last_row['close']
        if position == 'long':
            exit_price_adj = exit_price * (1 - SLIPPAGE)
            ret = (exit_price_adj - entry_price) / entry_price - COMMISSION
        else:
            exit_price_adj = exit_price * (1 + SLIPPAGE)
            ret = (entry_price - exit_price_adj) / entry_price - COMMISSION
        trade = {
            'tipo': position,
            'entrada': entry_price,
            'salida': exit_price_adj,
            'retorno': ret,
            'razon': 'fin_datos'
        }
        trades.append(trade)
        equity_curve.append(equity_curve[-1] + ret)
        if 'combo_id' in params:
            save_trade(params['combo_id'], trade, entry_time, df.index[-1], leverage, capital * leverage)
    
    # Métricas
    if trades:
        profits = [t['retorno'] for t in trades]
        total_profit = sum(profits)
        num_trades = len(trades)
        win_rate = sum(1 for p in profits if p > 0) / num_trades if num_trades > 0 else 0.0
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / (1 + running_max) if running_max.size > 0 else np.array([0.0])
        max_dd = drawdown.max() if len(drawdown) > 0 else 0.0
    else:
        total_profit = 0.0
        num_trades = 0
        win_rate = 0.0
        max_dd = 0.0
    
    metrics = {
        'profit': total_profit,
        'trades': num_trades,
        'win_rate': win_rate,
        'max_dd': max_dd
    }
    return trades, metrics

# ==================== OPTIMIZACIÓN CON ALMACENAMIENTO ====================
def optimizar_combinacion(dfs, tf_entrada, tf_tendencia):
    """Optimiza parámetros para una combinación y guarda resultados."""
    df_entrada = dfs[tf_entrada]
    df_tendencia = dfs[tf_tendencia]
    
    # Alinear fechas
    start = max(df_entrada.index[0], df_tendencia.index[0])
    end = min(df_entrada.index[-1], df_tendencia.index[-1])
    df_entrada = df_entrada.loc[start:end]
    df_tendencia = df_tendencia.loc[start:end]
    
    if len(df_entrada) < 50 or len(df_tendencia) < 10:
        print(f"  Datos insuficientes para {tf_entrada}/{tf_tendencia}")
        return None
    
    mejor_profit = -np.inf
    mejor_params = None
    mejores_metricas = None
    
    total = len(ADX_RANGE) * len(RSI_LOW_RANGE) * len(RSI_HIGH_RANGE) * len(MULT_STOP_RANGE) * len(MULT_TP_RANGE) * 2
    print(f"  Optimizando {tf_entrada}(entrada) / {tf_tendencia}(tendencia) - {total} combinaciones")
    
    count = 0
    for adx, rsi_low, rsi_high, mult_stop, mult_tp, use_slope in product(
            ADX_RANGE, RSI_LOW_RANGE, RSI_HIGH_RANGE, MULT_STOP_RANGE, MULT_TP_RANGE, [False, True]):
        if rsi_low >= rsi_high:
            continue
        count += 1
        if count % 100 == 0:
            print(f"    Progreso: {count}/{total}")
        
        params = {
            'adx_th': adx,
            'rsi_low': rsi_low,
            'rsi_high': rsi_high,
            'mult_stop': mult_stop,
            'mult_tp': mult_tp,
            'use_slope': use_slope
        }
        _, metrics = backtest_dual_mtf(df_entrada, df_tendencia, params, capital=1, leverage=1)
        
        # Guardar cada resultado en BD (opcional, pero puede ser muchos)
        # Mejor guardar solo el mejor
        if metrics['profit'] > mejor_profit:
            mejor_profit = metrics['profit']
            mejor_params = params.copy()
            mejores_metricas = metrics
    
    # Guardar mejor resultado
    if mejor_params:
        save_optimization_result(tf_entrada, tf_tendencia, mejor_params, mejores_metricas)
        print(f"    Mejor: profit={mejores_metricas['profit']:.4f}, trades={mejores_metricas['trades']}, WR={mejores_metricas['win_rate']*100:.1f}%")
    
    return mejor_params, mejores_metricas

# ==================== TRADING CON PARÁMETROS ÓPTIMOS ====================
def live_trading_loop():
    """Simula un bucle de trading en vivo usando los mejores parámetros de la BD."""
    init_database()
    # Obtener todas las combinaciones de timeframes con mejores parámetros
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT DISTINCT tf_entrada, tf_tendencia FROM optimization_results
    ''')
    combos = c.fetchall()
    conn.close()
    
    if not combos:
        print("No hay parámetros optimizados. Ejecuta primero la optimización.")
        return
    
    # Para cada combo, cargar mejores params
    params_dict = {}
    for tf_entrada, tf_tendencia in combos:
        params = get_best_params(tf_entrada, tf_tendencia)
        if params:
            params_dict[(tf_entrada, tf_tendencia)] = params
    
    # Descargar datos iniciales
    df_1m = fetch_klines(SYMBOL, INTERVAL_BASE, HOURS)  # Podrían ser menos horas para live
    dfs = {}
    for nombre, rule in TIMEFRAMES.items():
        dfs[nombre] = resample_ohlc(df_1m, rule)
    
    # Bucle principal (simulado aquí, en producción se haría cada minuto)
    while True:
        try:
            # Actualizar datos cada minuto (simulado)
            time.sleep(60)
            # En un sistema real, aquí se descargarían las nuevas velas y se evaluarían señales
            # Por simplicidad, mostramos los parámetros cargados
            print(f"\n[{datetime.now()}] Evaluando señales con parámetros óptimos...")
            for (tf_ent, tf_tend), params in params_dict.items():
                # Calcular apalancamiento dinámico basado en win_rate
                win_rate = params['win_rate']
                if win_rate > MIN_WIN_RATE_FOR_LEVERAGE:
                    leverage = min(MAX_LEVERAGE, int(MAX_LEVERAGE * (win_rate / 0.5)))
                else:
                    leverage = 1
                capital_por_operacion = BASE_CAPITAL * 0.02  # 2% del capital por trade
                print(f"    {tf_ent}/{tf_tend}: WR={win_rate*100:.1f}%, apalancamiento={leverage}")
            # Aquí iría la lógica de trading real
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

# ==================== PROGRAMA PRINCIPAL ====================
def main():
    import sys
    init_database()
    
    # Si se pasa argumento 'optimize', ejecuta optimización
    if len(sys.argv) > 1 and sys.argv[1] == 'optimize':
        print("=== Iniciando optimización ===")
        print("Descargando velas de 1 minuto...")
        df_1m = fetch_klines(SYMBOL, INTERVAL_BASE, HOURS)
        if df_1m.empty:
            print("Error al descargar datos.")
            return
        print(f"Velas descargadas: {len(df_1m)}")
        
        # Reagrupar
        dfs = {}
        for nombre, rule in TIMEFRAMES.items():
            dfs[nombre] = resample_ohlc(df_1m, rule)
            print(f"  {nombre}: {len(dfs[nombre])} velas")
        
        # Optimizar cada combinación
        for tf_entrada, tf_tendencia in TF_COMBINATIONS:
            print(f"\n--- Combinación: entrada {tf_entrada}, tendencia {tf_tendencia} ---")
            if tf_entrada not in dfs or tf_tendencia not in dfs:
                print("  Timeframe no disponible")
                continue
            start_time = time.time()
            optimizar_combinacion(dfs, tf_entrada, tf_tendencia)
            elapsed = time.time() - start_time
            print(f"  Tiempo: {elapsed:.1f}s")
        
        print("\nOptimización completada. Resultados guardados en la base de datos.")
    
    # Si se pasa argumento 'live', inicia bucle de trading
    elif len(sys.argv) > 1 and sys.argv[1] == 'live':
        print("=== Iniciando modo live (simulado) ===")
        live_trading_loop()
    
    else:
        # Modo por defecto: mostrar últimos resultados
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query('''
            SELECT tf_entrada, tf_tendencia, adx_th, rsi_low, rsi_high,
                   mult_stop, mult_tp, use_slope, profit, trades, win_rate, max_dd
            FROM optimization_results
            ORDER BY profit DESC
        ''', conn)
        conn.close()
        
        if df.empty:
            print("No hay resultados. Ejecuta con 'optimize' primero.")
        else:
            print("\n=== ÚLTIMOS RESULTADOS DE OPTIMIZACIÓN ===")
            print(df.to_string(index=False))
            
            # Mostrar mejores por combinación
            print("\n=== MEJORES POR COMBINACIÓN ===")
            best = df.loc[df.groupby(['tf_entrada', 'tf_tendencia'])['profit'].idxmax()]
            print(best.to_string(index=False))

if __name__ == "__main__":
    main()
