# skfin
Machine learning for portfolio management and trading

.. code:: python

    	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import StandardScaler
	from skfin import Ridge, MeanVariance, Backtester
	from skfin.datasets import load_kf_returns
	from skfin.plot import line

	estimator = make_pipeline(StandardScaler(with_mean=False), 
                          Ridge(), 
                          MeanVariance())

	returns_data = load_kf_returns(cache_dir='data')
	ret = returns_data['Monthly']['Average_Value_Weighted_Returns'][:'1999']

	transform_X = lambda x: x.rolling(12).mean().fillna(0).values
	transform_y = lambda x: x.shift(-1).values
	features = transform_X(ret)
	target = transform_y(ret)

	estimator = make_pipeline(StandardScaler(with_mean=False),
				  Ridge(), 
                          	  MeanVariance())

	bt = Backtester(estimator, ret).train(features, target)
	line(bt.pnl_, cumsum=True, title='Ridge')