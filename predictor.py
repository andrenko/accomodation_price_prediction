from collections import Counter
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

# Change pandas viewing options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Loading datasets
df_train = pd.read_csv('train.csv')
df_submission = pd.read_csv('submission.csv')
df = pd.concat([df_train, df_submission], keys=['train', 'submission'])
print(f"Shape of dataset: {df.shape}")
print(df.head())
print(df.info())

# ------------- Feature engineering ------------
# Converting price columns to numbers
print(f"Price column before conversion:\n{df['price'].head(2)}")
for column in ['price', 'cleaning_fee']:
    df[column] = df[column].apply(lambda x: float(str(x).replace(',', '').split('$')[-1]))
print(f"Price column after conversion:\n{df['price'].head(2)}")

# Converting percent columns to numbers
print(f"Percent columns before conversion:\n{df['host_response_rate'].unique(), df['host_acceptance_rate'].unique()}")
for column in ['host_response_rate', 'host_acceptance_rate']:
    df[column] = df[column].fillna(0).apply(lambda x: 0.01 * int(str(x).split('%')[0]))
print(f"Percent columns after conversion:\n{df['host_response_rate'].unique(), df['host_acceptance_rate'].unique()}")

# Converting boolean to numbers
print(f"Instant_bookable column before conversion:\n{df['instant_bookable'].head(2)}")
df['instant_bookable'] = df['instant_bookable'].apply(lambda x: 0 if x == 'f' else (1 if x == 't' else None))
print(f"Instant_bookable column after conversion:\n{df['instant_bookable'].head(2)}")

# Exploring object columns and dropping textual columns. We keep smart_location as location identifier because it
# contains city, state, and market values.
print(f"Object columns:\n{df.select_dtypes(object).head(2)}")
textual_columns_to_drop = ['name', 'summary', 'description', 'neighborhood_overview', 'transit', 'host_location',
                           'host_about', 'city', 'state', 'market', 'country_code', 'country']
df.drop(textual_columns_to_drop, axis=1, inplace=True)

# Check if host_neighbourhood do not contain values with slightly different spelling
host_neighbourhood_unique = df['host_neighbourhood'].dropna().unique()
host_neighbourhood_unique.sort()
print(f"Host_neighbourhood column unique values:\n{host_neighbourhood_unique}")

# As we can see Mount Pleasant and Mt. Pleasant is the same entity, so we replace Mt. Pleasant
df['host_neighbourhood'] = df['host_neighbourhood'].apply(lambda x: 'Mount Pleasant' if x == 'Mt. Pleasant' else x)
host_neighbourhood_unique = df['host_neighbourhood'].dropna().unique()
host_neighbourhood_unique.sort()
print(f"Host_neighbourhood column after processing:\n{host_neighbourhood_unique}")

# Check if neighbourhood_cleansed do not contain values with slightly different spelling
neighbourhood_cleansed_unique = df['neighbourhood_cleansed'].unique()
neighbourhood_cleansed_unique.sort()
print(f"Neighbourhood_cleansed column unique values:\n{neighbourhood_cleansed_unique}")

# Check if smart_location do not contain values with slightly different spelling
smart_location_unique = df['smart_location'].unique()
smart_location_unique.sort()
print(f"Smart_location column unique values:\n{smart_location_unique}")

# As we can see, there are three different spelling of 'Washington, DC', replace them with one
df['smart_location'] = df['smart_location'].apply(lambda x: 'Washington, DC' if x in ('Washington, D.C., DC',
                                                                                      'Washington , DC') else x)
smart_location_unique = df['smart_location'].unique()
smart_location_unique.sort()
print(f"Smart_location column after processing:\n{smart_location_unique}")

#  Now let's count locations. As we can see, there are a number of locations with small row counts. We will drop them.
print(f"Counting locations: \n{df['smart_location'].value_counts()}")
df.drop(df[df['smart_location'] != 'Washington, DC'].index, inplace=True)
print(f"Smart_location after dropping loctions with small row counts: \n{df['smart_location'].value_counts()}")
# When only one location left, we can drop entire column
df.drop('smart_location', axis=1, inplace=True)

# Check if property_type do not contain values with slightly different spelling
print(f"Property_type column unique values:\n{df['property_type'].unique()}")
# Check if room_type do not contain values with slightly different spelling
print(f"Room_type column unique values:\n{df['room_type'].unique()}")
# Check if bed_type do not contain values with slightly different spelling
print(f"Bed_type column unique values:\n{df['bed_type'].unique()}")

# Let's explore amenities
amenities = [amenity.strip('\"') for record in df['amenities'] for amenity in record[1:-1].split(',')]
print(f"Counting amenities:\n{Counter(amenities)}")
# We have 3013 rows in our dataset, as we can see, such amenities as 'Air Conditioning': 2883, 'Heating': 2863,
# 'Wireless Internet': 2855, 'Kitchen': 2785 are presented in almost every apartment, so we are interested in more
# distinctive features like 'Shampoo', 'Family/Kid Friendly', 'Carbon Monoxide Detector', 'Free Parking on Premises',
# 'Elevator in Building', 'Gym', 'Pets Allowed', 'Wheelchair Accessible', 'Breakfast', 'Pool', 'Smoking Allowed'.
# So now we are going to create new features for amenities and one feature for amenities_count
amenities = ['Shampoo', 'Family/Kid Friendly', 'Carbon Monoxide Detector', 'Free Parking on Premises', 'Breakfast',
             'Elevator in Building', 'Gym', 'Pets Allowed', 'Wheelchair Accessible', 'Pool', 'Smoking Allowed']
for amenity in amenities:
    df[amenity.lower()] = df['amenities'].apply(lambda x: int(amenity in x))
df['amenities_count'] = df.amenities.apply(lambda x: len(str(x)[1:].split(',')))
df.drop('amenities', axis=1, inplace=True)
print(f"New columns from amenities list:\n{df.loc[:, 'shampoo':'amenities_count'].head()}")

# Now let's look at host_since column. We can treat these values as host experience, so we can calculate difference
# between today date and host_since date and save it as number
print(f"Null data: {df['host_since'].isna().sum()}")
df['host_experience'] = df['host_since'].apply(lambda x: int(str(date.today() - date.fromisoformat(x)).split(' ')[0]))
df.drop('host_since', axis=1, inplace=True)
print(f"Host experience column:\n{df['host_experience'].head(2)}")

# Now let's count the number of host verifications methods and save it as host_verifications_count column
df['host_verifications_count'] = df['host_verifications'].apply(lambda x: len(x.split(', ')))
df.drop('host_verifications', axis=1, inplace=True)
print(f"Host verifications count column:\n{df['host_verifications_count'].head(2)}")

# ------------- Dealing with missing values ------------
print(f"Missing values count for the whole dataset:\n{df.isnull().sum().sort_values(ascending=False).head(20)}")
# Columns bathrooms, bedrooms, beds, zipcode, property_type contain small number of nan values, so we can just drop
# these rows
df.dropna(axis=0, how='any', subset=['bathrooms', 'bedrooms', 'beds', 'zipcode', 'property_type'], inplace=True)
print(df.isnull().sum().sort_values(ascending=False).head(20))
# As we can see, there are 600+ rows that does not contain review scores, we will make one-hot encoding for
# review_scores_rating with special column for 'no reviews', while dropping other review scores columns. We will also
# fill nan with 0 for reviews_per_month column.
plt.hist(df['review_scores_rating'][~df['review_scores_rating'].isnull()])
plt.title("Histogram of Review Scores Ratings")
plt.xlabel("Review Score")
plt.ylabel("Frequency")
plt.show()
print("Counting different scores in review_scores_rating column:\n",
      df['review_scores_rating'].value_counts().sort_index(ascending=False).head(15))
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df['review_scores_rating'] = df['review_scores_rating'].fillna('no rating')


def group_review_scores(score):
    if score == 'no rating':
        return 'no rating'
    elif score > 98:
        return '99-100'
    elif score > 96:
        return '97-98'
    elif score > 94:
        return '95-96'
    elif score > 92:
        return '93-94'
    elif score > 90:
        return '91-92'
    elif score > 88:
        return '89-90'
    elif score > 86:
        return '87-88'
    elif score > 84:
        return '85-86'
    elif score > 82:
        return '83-84'
    elif score > 75:
        return '76-82'
    elif score > 69:
        return '70-75'
    elif score > 59:
        return '60-69'
    elif score > 44:
        return '45-59'
    elif score > 0:
        return '0-44'


df['review_scores_rating'] = df['review_scores_rating'].apply(group_review_scores)
one_hot_review = pd.get_dummies(df['review_scores_rating'], prefix='review rating:', prefix_sep=" ")
df = df.join(one_hot_review)
review_columns_to_drop = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                          'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                          'review_scores_value']
df.drop(review_columns_to_drop, axis=1, inplace=True)
print(f"One-hot encoded review rating columns:\n{df.loc[:, 'review rating: 0-44':].head(5)}")

# Let's look at price and cleaning fee columns. We will drop outliers and explore correlation between them
print(f"Price frequency:\n{df['price'].value_counts().sort_index(ascending=False).head()}")
df.drop(df[df['price'] > 1500].index, inplace=True)
print(f"Price frequency after dropping top values:\n{df['price'].value_counts().sort_index(ascending=False).head()}")
print(f"Cleaning fee frequency:\n{df['cleaning_fee'].value_counts().sort_index(ascending=False).head()}")
plt.title("Cleaning fee vs price")
sns.lineplot(x='price', y='cleaning_fee', data=df[['price', 'cleaning_fee']])
plt.xlabel('Price', fontsize=11)
plt.ylabel('Cleaning fee', fontsize=11)
plt.show()

# As we can see, the correlation between cleaning fee and the price in non-linear, with many outliers. So we will
# interpolate cleaning fee curve in order to fill its missing values.
plt.title("Cleaning fee with missing values")
df['cleaning_fee'].plot().set_xlim(0, 100)
plt.xlabel('Cleaning fee', fontsize=11)
plt.show()

plt.title("Interpolated cleaning fee")
df['cleaning_fee'].interpolate().plot().set_xlim(0, 100)
plt.xlabel('Cleaning fee', fontsize=11)
plt.show()

df['cleaning_fee'] = df['cleaning_fee'].interpolate()
df.dropna(axis=0, how='any', subset=['cleaning_fee'], inplace=True)
print(f"Left missing values :\n{df.isnull().sum().sort_values(ascending=False).head()}")

# Finally we will drop nan host_neighbourhood values
df.dropna(axis=0, how='any', subset=['host_neighbourhood'], inplace=True)
print(f"Left missing values :\n{df.isnull().sum().sort_values(ascending=False).head()}")

# Before proceeding, we have to convert categorical columns
for column in ['host_neighbourhood', 'neighbourhood_cleansed', 'zipcode', 'property_type',
               'room_type', 'bed_type', 'cancellation_policy']:
    df[column] = df[column].astype('category')
    df[column] = df[column].cat.codes
print(f"Final shape of dataset: {df.shape}")
print(df.head())
print(df.info())

# ----------------- Building regressor -------------------
y = df['price'].loc['train']
X = df.drop(['price'], axis=1).loc['train']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train dataset shape: {X_train.shape, y_train.shape}")
print(f"Test dataset shape: {X_test.shape, y_test.shape}")

models = [LinearRegression(),
          GradientBoostingRegressor(),
          RandomForestRegressor(),
          KNeighborsRegressor(),
          SVR(),
          LogisticRegression(),
          Ridge(),
          Lasso(),
          ElasticNet()]

r2_scores = {}
for model in models:
    model.fit(X_train, y_train)
    r2_scores[str(model)] = (r2_score(y_test, model.predict(X_test)))

# Plot r2 scores
bars = plt.bar(range(len(r2_scores)), list(r2_scores.values()), align='center')
plt.xticks(range(len(r2_scores)), list(r2_scores.keys()), rotation=90)
for bar in bars:
    yval = round(bar.get_height(), 3)
    plt.text(bar.get_x(), yval + .005, yval)
plt.title('R2 scores for different regressors')
plt.show()

# Trying to optimize Gradient Boosting regressor by running GridSearchCV
params_gradient_boosting = {'learning_rate': [0.1, 0.07, 0.05],
                            'max_depth': [4, 5, 6],
                            'min_samples_leaf': [3, 5, 9],
                            'max_features': [0.2, 0.1, 0.01],
                            'n_estimators': [100, 250, 500]}
gradient_boosting = GradientBoostingRegressor(random_state=42)
gs = GridSearchCV(gradient_boosting, params_gradient_boosting, cv=3, n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print(f"The best estimator across ALL searched params:\n{gs.best_estimator_}")
print(f"The best score across ALL searched params:\n{gs.best_score_}")
print(f"The best parameters across ALL searched params:\n{gs.best_params_}")
gradient_boosting_with_gs_params = GradientBoostingRegressor(**gs.best_params_)
gradient_boosting_with_gs_params.fit(X_train, y_train)
print(f"R2 score for GradientBoostingRegressor optimized by GridSearchCV:\n",
      r2_score(y_test, gradient_boosting_with_gs_params.predict(X_test)))

# GridSearch did not show better result, so will stick with default parameters
gradient_boosting.fit(X_train, y_train)
predicted = gradient_boosting.predict(X_test)
print(f"R2 score for Gradient Boosting with default parameters:\n{r2_score(y_test, predicted)}")

# Plot predicted vs actual price
plt.style.use('ggplot')
plt.plot(y_test, predicted, 'b.')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Gradient Boosting Regressor')
plt.show()

y_train_pred = gradient_boosting.predict(X_train)
y_test_pred = gradient_boosting.predict(X_test)

print(f'MAE train: {mean_absolute_error(y_train, y_train_pred):.3f}, '
      f'test: {mean_absolute_error(y_test, y_test_pred):.3f}')
print(f'R2 score train: {r2_score(y_train, y_train_pred):.3f}, test: {r2_score(y_test, y_test_pred):.3f}')

# Plot residuals
plt.scatter(y_train_pred, y_train_pred-y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred-y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=y_test_pred.min() - 50, xmax=y_test_pred.max() + 50, lw=2, color='red')
plt.xlim([y_test_pred.min() - 50, y_test_pred.max() + 50])
plt.show()

# Plot feature importance
pd.Series(gradient_boosting.feature_importances_, index=X.columns).nlargest(20).plot(kind='barh').invert_yaxis()
plt.xlabel('Relative importance')
plt.title('Feature importance')
plt.show()

# The distribution of predicted prices
plt.hist(gradient_boosting.predict(X))
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Predicted Housing Prices (fitted values)')
plt.show()

# Plotting predicted prices vs actual prices:
plt.scatter(y, gradient_boosting.predict(X))
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs Predicted Prices")
plt.show()

# Finally we will predict price for submission dataset and save the result
X_df_submission = df.drop(['price'], axis=1).loc['submission']
df_submission['price_prediction'] = pd.Series(gradient_boosting.predict(X_df_submission), index=X_df_submission.index)
df_submission.to_csv('submission_with_predicted_price.csv')
