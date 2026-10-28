# Pandas groupby

groupby splits a DataFrame into groups by one or more keys, applies a function to each group, and combines the results (split-apply-combine). Common aggregations are sum, mean, count and size. count ignores NaN values while size counts all rows.

agg lets you apply several functions at once, for example df.groupby("city").agg(total=("sales", "sum"), avg=("sales", "mean")). transform returns a result aligned with the original index, which is useful for adding a per-group mean as a new column. apply is the most flexible but slowest option.

# Pandas merge

merge joins two DataFrames on key columns, like SQL joins. The how argument can be inner, left, right or outer. An inner merge keeps only keys present in both frames; a left merge keeps every row of the left frame and fills missing matches with NaN.

Duplicate keys on both sides produce a many-to-many join that multiplies rows. Use the validate argument, for example validate="one_to_one", to catch unexpected duplicates early. Use indicator=True to see which side each row came from.
