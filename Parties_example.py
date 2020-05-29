# EDA
# iris dataset
pd.plotting.scatter_matrix(df, 
                           c=target_column,  # c stands for color
                           fig_size=[8, 8],  
                           s=150,  # shape (number of rows)
                           marker='D'
                          )

# Demacrat-republican classification example
# since all the features in this example are binary, scatter_matrix is not very useful. Instead we use countplot
plt.figure()
sns.countplot(x='predictor_col', hue='target_col', data=df, palette='RdBu') # RdBU means: red blue
plt.xticks([0,1], ['No', 'Yes'])  # mapping 0,1 to 'No', 'Yes'
plt.show()
