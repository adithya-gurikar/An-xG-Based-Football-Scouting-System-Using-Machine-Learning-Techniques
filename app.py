import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

app = Flask(__name__)

dfn = pd.read_csv('Football_Scouts_Database_Raw_Stats_Normalised_Floats.csv')
dfn = dfn.fillna(0)

Centre_forward_qualities = ['Goals', 'Goals/90']
Winger_qualities = ['Progressive Carries', 'Successful Take Ons', 'Touches in Attacking 3rd',
                    'Successful Take Ons/90', 'Touches in Attacking 3rd/90', 'Progressive Carries/90']
Attacking_mid_qualities = ['Key Passes', 'Key Passes/90']
Central_mid_qualities = ['Progressive Passes', 'Passes Completed', 'Progressive Passes/90', 'Passes Completed/90']
Defensive_mid_qualities = ['Tackles Won', 'Interceptions', 'Ball Recoveries', 'Tackles Won/90', 'Interceptions/90', 'Ball Recoveries/90']
Wingback_qualities = ['Tackles Won', 'Crosses into penalty area', 'Tackles Won/90', 'Crosses into penalty area/90']
Ballplaying_def_qualities = ['% of Aerial Duels won', 'Shots Blocked', 'Clearances', 'Passes Completed', 'Shots Blocked/90', 'Clearances/90', 'Passes Completed/90']
Defensive_cb_qualities = ['Shots Blocked', 'Clearances', '% of Aerial Duels won', 'Shots Blocked/90', 'Clearances/90']

y_forwards = dfn[Centre_forward_qualities]
y_midfielders = dfn[Winger_qualities + Attacking_mid_qualities + Central_mid_qualities + Defensive_mid_qualities]
y_defenders = dfn[Wingback_qualities + Ballplaying_def_qualities + Defensive_cb_qualities]

X = dfn.drop(['Player', 'Position'], axis=1)
X = X.drop(columns=y_forwards)

pca = PCA(n_components=30)
pca.fit(X)

X_train, X_test_forwards, y_train_forwards, y_test_forwards = train_test_split(X, y_forwards, test_size=0.2)
X_train, X_test_midfielders, y_train_midfielders, y_test_midfielders = train_test_split(X, y_midfielders, test_size=0.2)
X_train, X_test_defenders, y_train_defenders, y_test_defenders = train_test_split(X, y_defenders, test_size=0.2)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'Lasso': Lasso(),
    'Ridge': Ridge()
}

for name, model in models.items():
    model.fit(X_train, y_train_forwards)

def get_results(position, y_test, X_test):
    target_variable = None
    if position == 'forwards':
        target_variable = 'Non-Penalty(xG-Goals)'
    elif position == 'midfielders':
        target_variable = 'npxGi-npGi'
    elif position == 'defenders':
        target_variable = 'xAG-Assists'
    else:
        return []

    r2_values = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name}: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
        r2_values.append(r2)

    best_model_index = r2_values.index(max(r2_values))

    if best_model_index == 0:
        model = models['Linear Regression']
    elif best_model_index == 1:
        model = models['Random Forest']
    elif best_model_index == 2:
        model = models['XGBoost']
    elif best_model_index == 3:
        model = models['Lasso']
    else:
        model = models['Ridge']

    model.fit(X, y_forwards)
    coefficients = model.coef_
    weighted_sum = np.dot(X, coefficients.T).sum(axis=1)
    dfn['weighted_sum'] = weighted_sum
    dfn_combined = dfn.groupby('Player').agg({'weighted_sum': 'mean'})
    xg_df = pd.read_csv('Football_Scouts_Database_xG.csv')
    xg_df_filtered = xg_df[xg_df['Position'].str.contains(position.upper())]
    xg_df_filtered.loc[:, 'weighted_sum'] = xg_df_filtered['Unnamed: 0'].map(dfn_combined['weighted_sum'])
    xg_df_sorted = xg_df_filtered.sort_values(by='weighted_sum', ascending=False)

    xg_df_sorted_numeric = xg_df_sorted.select_dtypes(include='number')
    top_10_avg = xg_df_sorted_numeric.head(10).mean()

    euclidean_distances = cdist(xg_df_sorted_numeric, top_10_avg.to_frame().T, metric='minkowski', p=2)
    manhattan_distances = cdist(xg_df_sorted_numeric, top_10_avg.to_frame().T, metric='minkowski', p=1)
    xg_df_sorted['Euclidean Distance'] = euclidean_distances[:, 0]
    xg_df_sorted['Manhattan Distance'] = manhattan_distances[:, 0]

    xg_df_sorted['Euclidean Distance (Normalized)'] = 1 - (xg_df_sorted['Euclidean Distance'] - xg_df_sorted['Euclidean Distance'].min()) / (xg_df_sorted['Euclidean Distance'].max() - xg_df_sorted['Euclidean Distance'].min())
    xg_df_sorted['Manhattan Distance (Normalized)'] = 1 - (xg_df_sorted['Manhattan Distance'] - xg_df_sorted['Manhattan Distance'].min()) / (xg_df_sorted['Manhattan Distance'].max() - xg_df_sorted['Manhattan Distance'].min())
    xg_df_sorted['Target Variable (Normalized)'] = (xg_df_sorted[target_variable] - xg_df_sorted[target_variable].min()) / (xg_df_sorted[target_variable].max() - xg_df_sorted[target_variable].min())

    weights = [0.25, 0.25, 0.5]
    xg_df_sorted['Combined Score'] = xg_df_sorted[['Euclidean Distance (Normalized)', 'Manhattan Distance (Normalized)', 'Target Variable (Normalized)']].dot(weights)

    sorted_df = xg_df_sorted.sort_values(by='Combined Score', ascending=False)

    return sorted_df.head(20).to_dict(orient='records')

@app.route('/get_sorted_df', methods=['POST'])
def get_sorted_df():
    data = request.get_json()
    position = data['position']

    if position == 'forwards':
        y_test = y_test_forwards
        X_test = X_test_forwards
    elif position == 'midfielders':
        y_test = y_test_midfielders
        X_test = X_test_midfielders
    elif position == 'defenders':
        y_test = y_test_defenders
        X_test = X_test_defenders
    else:
        return jsonify([])

    results = get_results(position, y_test, X_test)
    return jsonify(results)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Football Players Ranking</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }

            .container {
                max-width: 800px;
                margin: auto;
            }

            .btn {
                margin-top: 10px;
            }

            #results {
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4">Football Players Ranking</h1>
            <div class="form-group mt-4">
                <label for="positionSelect">Select Position:</label>
                <select class="form-control" id="positionSelect">
                    <option value="forwards">Forwards</option>
                    <option value="midfielders">Midfielders</option>
                    <option value="defenders">Defenders</option>
                </select>
            </div>
            <button class="btn btn-primary" onclick="getResults()">Show Results</button>
            <div class="mt-4" id="results"></div>
        </div>

        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
            function getResults() {
                const position = $('#positionSelect').val();
                $.ajax({
                    type: 'POST',
                    url: '/get_sorted_df',
                    data: JSON.stringify({ 'position': position }),
                    contentType: 'application/json',
                    success: function (data) {
                        displayResults(data);
                    },
                    error: function (error) {
                        console.error('Error:', error);
                    }
                });
            }

            function displayResults(results) {
                const resultsDiv = $('#results');
                resultsDiv.empty();
                const table = $('<table class="table table-bordered table-striped"></table>');
                const headerRow = $('<tr><th>Player</th><th>Position</th><th>Weighted Sum</th></tr>');
                table.append(headerRow);
                for (const player of results) {
                    const row = $(`<tr><td>${player['Player']}</td><td>${player['Position']}</td><td>${player['weighted_sum']}</td></tr>`);
                    table.append(row);
                }
                resultsDiv.append(table);
            }
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
