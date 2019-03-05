# Amazon_foodreview_summarization

Data Source - https://www.kaggle.com/snap/amazon-fine-food-reviews

**Dataset Overview**

The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.

- Number of reviews: 568,454
- Number of users: 256,059
- Number of products: 74,258
- Timespan: Oct 1999 - Oct 2012
- Number of Attributes/Columns in data: 10

Attribute Information:

1. Id
2. ProductId - unique identifier for the product
3. UserId - unqiue identifier for the user
4. ProfileName
5. HelpfulnessNumerator - number of users who found the review helpful
6. HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
7. Score - rating between 1 and 5
8. Time - timestamp for the review
9. Summary - brief summary of the review
10.Text - text of the review


**Objective**

Given a product, summarize the review based on the ratings. Can be done in two ways
1. Classify as positive (rating 5 & 4) or negative (rating 1 & 2) and summarize them
2. Summarize individual rating wise


**[Q]** How to determine if a review is positive or negative?

**[Ans]** We could use the Score/Rating. A rating of 4 or 5 could be cosnidered a positive review. A review of 1 or 2 could be considered negative. A review of 3 is nuetral and ignored. This is an approximate and proxy way of determining the polarity (positivity/negativity) of a review.



















