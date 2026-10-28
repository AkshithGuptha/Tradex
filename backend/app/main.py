from __future__ import annotations
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI,File,HTTPException,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
app=FastAPI(title='Tradex API',version='1.0.0',description='Simulation-only RL trading research API')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
DATA_PATH=Path(__file__).resolve().parent/'sample_market.csv';market_df=None;agent=None;last_result=None;trade_log=[]
def sample_data(n=320,seed=23):
 rng=np.random.default_rng(seed);dates=pd.bdate_range(end=pd.Timestamp.today().normalize(),periods=n);ret=rng.normal(.00035,.012,n);ret[65:105]+=.001;ret[170:215]-=.0006;close=100*np.exp(np.cumsum(ret));op=np.r_[close[0],close[:-1]]*(1+rng.normal(0,.002,n));hi=np.maximum(op,close)*(1+rng.uniform(0,.009,n));lo=np.minimum(op,close)*(1-rng.uniform(0,.009,n));vol=rng.integers(500000,3000000,n);return pd.DataFrame({'date':dates,'open':op,'high':hi,'low':lo,'close':close,'volume':vol})
def get_market():
 global market_df
 if market_df is None:market_df=pd.read_csv(DATA_PATH,parse_dates=['date']) if DATA_PATH.exists() and DATA_PATH.stat().st_size>40 else sample_data()
 return market_df.copy()
def features(df):
 d=df.copy().sort_values('date').reset_index(drop=True);d['return']=d.close.pct_change().fillna(0);d['sma_fast']=d.close.rolling(5,min_periods=1).mean();d['sma_slow']=d.close.rolling(20,min_periods=1).mean();d['momentum']=d.close.pct_change(5).fillna(0);d['volatility']=d['return'].rolling(10,min_periods=2).std().fillna(0);d['trend']=(d.sma_fast>d.sma_slow).astype(int);d['mom_bin']=pd.cut(d.momentum,[-np.inf,-.025,-.005,.005,.025,np.inf],labels=False).fillna(2).astype(int);d['vol_bin']=np.digitize(d.volatility,d.volatility.quantile([.33,.66]).to_numpy());d['state']=list(zip(d.trend.astype(int),d.mom_bin.astype(int),d.vol_bin.astype(int)));return d
class TrainRequest(BaseModel):episodes:int=Field(180,ge=10,le=2000);learning_rate:float=Field(.12,ge=.01,le=1);discount:float=Field(.92,ge=0,le=.999);transaction_cost_bps:float=Field(5,ge=0,le=100)
class BacktestRequest(BaseModel):transaction_cost_bps:float=Field(5,ge=0,le=100);initial_cash:float=Field(10000,gt=0,le=100000000)
class Agent:
 def __init__(self,a,g,c):self.q={};self.alpha=a;self.gamma=g;self.cost=c/10000;self.epsilon=1.
 def row(self,s):
  if s not in self.q:self.q[s]=np.zeros(3)
  return self.q[s]
 def choose(self,s,explore=False):
  if explore and np.random.random()<self.epsilon:return int(np.random.randint(0,3))
  return int(np.argmax(self.row(s)))
 def train(self,d,episodes):
  rewards=[];states=d.state.tolist();rets=d['return'].to_numpy()
  for _ in range(episodes):
   total=0.
   for i in range(len(d)-1):
    s=states[i];a=self.choose(s,True);reward=(float(rets[i+1])-self.cost if a==1 else (-self.cost if a==2 else 0))-0.15*float(d.volatility.iloc[i]);row=self.row(s);row[a]+=self.alpha*(reward+self.gamma*float(np.max(self.row(states[i+1])))-row[a]);total+=reward
   self.epsilon=max(.04,self.epsilon*.985);rewards.append(total)
  return rewards
def calc_metrics(eq,dr,trades):
 e=np.asarray(eq);r=np.asarray(dr);dd=e/np.maximum.accumulate(e)-1;closed=[x for x in trades if x['type']=='SELL'];wins=sum(x.get('pnl',0)>0 for x in closed);return {'total_return':round(float((e[-1]/e[0]-1)*100),2),'max_drawdown':round(float(abs(dd.min())*100),2),'annualized_volatility':round(float(np.std(r)*np.sqrt(252)*100),2),'sharpe_ratio':round(float(np.mean(r)/(np.std(r)+1e-12)*np.sqrt(252)),2),'win_rate':round(wins/len(closed)*100 if closed else 0,2)}
@app.get('/api/health')
def health():return {'status':'ok','service':'Tradex API','mode':'simulation-only'}
@app.get('/api/market')
def market():
 d=features(get_market());return {'symbol':'TRDX-SIM','source':'generated or uploaded sample data','data':[{'date':x.date.strftime('%Y-%m-%d'),'open':round(float(x.open),2),'high':round(float(x.high),2),'low':round(float(x.low),2),'close':round(float(x.close),2),'volume':int(x.volume),'return':round(float(x['return'])*100,3)} for _,x in d.tail(180).iterrows()]}
@app.post('/api/market/upload')
async def upload_market(file:UploadFile=File(...)):
 global market_df
 if not file.filename or not file.filename.lower().endswith('.csv'):raise HTTPException(400,'Upload a CSV file.')
 try:
  df=pd.read_csv(StringIO((await file.read()).decode('utf-8-sig')));df.columns=[x.strip().lower() for x in df.columns]
  if 'close' not in df:raise ValueError('CSV must contain close column')
  if 'date' not in df:df['date']=pd.bdate_range(end=pd.Timestamp.today().normalize(),periods=len(df))
  df['date']=pd.to_datetime(df.date,errors='raise')
  for col in ['open','high','low']:
   if col not in df:df[col]=df.close
  if 'volume' not in df:df['volume']=0
  for col in ['open','high','low','close','volume']:df[col]=pd.to_numeric(df[col],errors='raise')
  df=df[['date','open','high','low','close','volume']].dropna().sort_values('date')
  if len(df)<30:raise ValueError('Provide at least 30 valid rows')
  market_df=df.reset_index(drop=True);return {'message':'Market data loaded for this API session','rows':len(df)}
 except Exception as e:raise HTTPException(400,f'Could not parse CSV: {e}') from e
@app.post('/api/train')
def train(req:TrainRequest):
 global agent
 d=features(get_market());agent=Agent(req.learning_rate,req.discount,req.transaction_cost_bps);rewards=agent.train(d,req.episodes);return {'status':'trained','episodes':req.episodes,'states_learned':len(agent.q),'epsilon':round(agent.epsilon,3),'reward_history':[{'episode':i+1,'reward':round(v*100,3)} for i,v in enumerate(rewards)],'message':'Simulation training complete; not a forecast.'}
@app.post('/api/backtest')
def backtest(req:BacktestRequest):
 global last_result,trade_log
 d=features(get_market());a=agent
 if a is None:a=Agent(.12,.92,req.transaction_cost_bps);a.train(d,120)
 a.cost=req.transaction_cost_bps/10000;cash=float(req.initial_cash);units=0.;entry=0.;eq=[];dr=[];tr=[];curve=[];prev=cash
 for _,row in d.iterrows():
  price=float(row.close);action=a.choose(row.state)
  if action==1 and units==0 and cash>0:
   units=cash/price;entry=price;cash=max(0,cash-cash*a.cost);tr.append({'date':row.date.strftime('%Y-%m-%d'),'type':'BUY','price':round(price,2),'units':round(units,6),'pnl':0.})
  elif action==2 and units>0:
   proceeds=units*price;pnl=(price-entry)*units-proceeds*a.cost;cash=proceeds*(1-a.cost);tr.append({'date':row.date.strftime('%Y-%m-%d'),'type':'SELL','price':round(price,2),'units':round(units,6),'pnl':round(pnl,2)});units=0.;entry=0.
  value=cash+units*price;eq.append(value);dr.append(value/prev-1 if prev else 0);prev=value;curve.append({'date':row.date.strftime('%Y-%m-%d'),'equity':round(value,2),'close':round(price,2)})
 trade_log=list(reversed(tr[-50:]));last_result={'initial_cash':req.initial_cash,'final_value':round(eq[-1],2),'cash':round(cash,2),'position_value':round(units*float(d.iloc[-1].close),2),'equity_curve':curve[-180:],'trades':list(reversed(tr[-50:])),'metrics':calc_metrics(eq,dr,tr)};return last_result
@app.get('/api/trades')
def trades():return {'trades':trade_log,'count':len(trade_log)}
@app.get('/api/portfolio')
def portfolio():return {'ready':False,'message':'Run a backtest to generate analytics'} if last_result is None else {'ready':True,**last_result}
