import pandas as pd
import json
from datetime import datetime
import datetime
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.offline as pltoff
import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import pointplot
from seaborn import lineplot

# please note no exception handling was added since this is a once off analysis and not
# a production application


def data_preparation():
    df = pd.read_json('/Users/iliaspsi/PycharmProjects/efood_assesment/exports/bq-results-orders.json',lines=True)
    df['submit_dt'] = df['submit_dt'].str.replace(r'UTC', '') #regular exp, remove UTC string
    df['submit_dt'] = pd.to_datetime(df['submit_dt'], format='%Y-%m-%d %H:%M:%S')
    df["submit_dt"] = df["submit_dt"].dt.date

    snapshot_date = max(df.submit_dt) + datetime.timedelta(days=1)
    print(snapshot_date);

    customers = df.groupby(['user_id']).agg({
        'submit_dt': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'count',
        'basket': 'sum'})

    customers.rename(columns = {'submit_dt': 'Recency',
                            'order_id': 'Frequency',
                            'basket': 'MonetaryValue'}, inplace=True)
    return customers


def normalize_data(customers):
    scaler = StandardScaler();
    scaler.fit(customers)
    customers_normalized = scaler.transform(customers)
    print(np.shape(customers_normalized)[0])
    return customers_normalized


def decide_numof_clusters(customers_normalized):
    sse = {}
    for k in range(1, 11):
         kmeans = KMeans(n_clusters=k, random_state=42)
         kmeans.fit(customers_normalized)
         sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

    trace1 = {
     "name": "Elbow curve for k means mini-batch clustering",
     "type": "scatter",
     "x": list(sse.keys()),
     "y": list(sse.values())
    }
    data = [trace1]
    layout = {
     "title": "Elbow curve for k means mini-batch clustering",
     "xaxis": {
        "title": "K value",
        "titlefont": {
                  "size": 18,
                  "color": "#7f7f7f",
                  "family": "Courier New, monospace"
                     }
    },
     "yaxis": {
         "title": "sum of squared errors",
         "titlefont": {
                   "size": 18,
                   "color": "#7f7f7f",
                   "family": "Courier New, monospace"
                    }
    }
    }
    fig = dict(data=data, layout=layout)
    pltoff.offline.plot(fig,filename='elbow_curve')


def apply_kmeans(customers_normalized, customers):
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(customers_normalized)
    model.labels_.shape

    customers["Cluster"] = model.labels_
    segments = customers.groupby('Cluster').agg({
                'Cluster': 'max',
                'Recency':'mean',
                'Frequency':'mean',
                'MonetaryValue':['mean', 'count']}).round(2)

    print(segments)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(segments.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[segments.Cluster,segments.Recency, segments.Frequency, segments.MonetaryValue['mean'], segments.MonetaryValue['count']],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.show()
    return model


def present_data(customers_normalized,model):
    # Create the dataframe
    df_normalized = pd.DataFrame(customers_normalized, columns=['Recency', 'Frequency', 'MonetaryValue'])
    df_normalized['ID'] = customers.index
    df_normalized['Cluster'] = model.labels_
    # Melt The Data
    df_nor_melt = pd.melt(df_normalized.reset_index(),
                         id_vars=['ID', 'Cluster'],
                         value_vars=['Frequency','MonetaryValue','Recency'],
                        var_name='Attribute',
                        value_name='Value')
    df_nor_melt.head()
    # Visualize it
    lineplot(x='Attribute', y='Value', hue='Cluster', data=df_nor_melt)
    plt.show()


def scatter_plot3d(customers_normalized,model):
    df_normalized = pd.DataFrame(customers_normalized, columns=['Recency', 'Frequency', 'MonetaryValue'])
    df_normalized['ID'] = customers.index
    df_normalized['Cluster'] = model.labels_

    fig = px.scatter_3d(df_normalized, x='Recency', y='Frequency', z='MonetaryValue',
                        color='Cluster',
                        title="3D Scatter Plot")
    fig.show()


if __name__ == "__main__":
    # execute only if run as a script
    customers = data_preparation()
    customers_normalized = normalize_data(customers)
    decide_numof_clusters(customers_normalized)
    model = apply_kmeans(customers_normalized, customers)
    present_data(customers_normalized,model)
    scatter_plot3d(customers_normalized,model)
