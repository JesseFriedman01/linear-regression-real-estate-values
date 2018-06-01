import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime,date
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

########################################################################
# chartstringLR(dateio, zip_code)
# Purpose: use linear regression based on historical real estate values
#          in zip_code to predict future value on date_io, plot
#          values/data on graph
# Pre-conditions: dateio is a string in yyyy-m-d format, zip_code is a
#                 valid 5 digit zip code in which historical data exists
# Post-conditions: None
########################################################################
def chartstringLR(fut_date, zip_code):

    # https://www.quandl.com/data/ZILLOW-Zillow-Real-Estate-Research
    # Zillow Home Value Index - All Homes - zip_code
    location_str = "ZILLOW/z%s_ZHVIAH" % ( str(zip_code) )

    df = quandl.get(location_str)

    dateraw = df.index

    datetoordinal = []

    # convert dates in dateraw to list of toordinals because they're easier to work with in this format
    for i in range(len(dateraw)):
        strdate = dateraw[i].strftime("%m%d%y")
        datetoordinal.append(datetime.strptime(strdate, '%m%d%y').toordinal())

    X = np.array(datetoordinal).reshape(-1, 1)
    y = np.array(df['Value']).reshape(-1, 1)

    # produces y-coordinates for regression line via y = mx + b
    def line_coords(slope, y_int, dates):
        y_coor = []
        for date in dates:
            y_coor.append((float(slope[0][0]) * date) + float(y_int[0]))
        return (y_coor)

    X_fut = datetoordinal

    # extend "future prediction line"
    for i in range(100):
        X_fut.append(X_fut[-1] + i)

    # toordinal dates back to regular date format
    X_fut_regular_dates = []
    for i in range(len(X_fut)):
        X_fut_regular_dates.append(date.fromordinal(X_fut[i]))

    # 20% of sample reserved for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    lr = LinearRegression().fit(X_train, y_train)

    a = line_coords(lr.coef_, lr.intercept_, X_fut)

    #confidence=(lr.score(X_test, y_test)) # commented out but can be used as needed

    pred_date = datetime.strptime(fut_date, '%Y-%m-%d').date()
    pred_date = pred_date.toordinal()

    pred = (lr.predict(pred_date))
    pred_lr = lr.predict(X)

    fig, ax = plt.subplots()

    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)

    for spine in plt.gca().spines.values():
        spine.set_alpha(.1)

    ax.plot(df.index, y)
    ax.plot(df.index, pred_lr, '--')
    ax.plot(X_fut_regular_dates, a, '--', color='g', alpha=.25)

    ax.scatter(pred_date, pred, color='g', s=100)

    # percent change of predicted value versus today's (linear regression) value
    pct_change = (pred[0][0] - int(y[-1])) / int(y[-1]) * 100

    annotate_string = "Prediction Data:\n\nDate: %s \nEst. Value: %s \nPct. Change: %s%%" % (
    fut_date, '${:,.2f}'.format(pred[0][0]), round(pct_change, 2))

    # rectangle with data on bottom right of chart
    at = AnchoredText(annotate_string,
                      prop=dict(size=10, color='darkgreen'), frameon=True,
                      loc=4
                      )
    at.patch.set_boxstyle("round")
    at.patch.set_fc('whitesmoke')
    at.patch.set_alpha(.25)

    ax.add_artist(at)

    plt.xlabel('Date')
    plt.ylabel('Value')

    plt.legend([ str(zip_code) + ' Value Index', 'Regression Line', 'Future Prediction Line'])

    plt.show()

nyc = chartstringLR( '2030-1-1', 10012 )






