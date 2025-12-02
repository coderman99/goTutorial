import pandas as pd
from labelling import label_business_cycle

# Fake macro monthly data (just to test the merge)
macro = pd.DataFrame({
    "name": ["Test"] * 5,
    "value": [1, 2, 3, 4, 5]
}, index=pd.to_datetime([
    "2010-01-31", "2010-02-28", "2010-03-31", "2010-04-30", "2010-05-31"
]))

# Fake S&P500 monthly
spx = pd.DataFrame({
    "sp500": [1123, 1089, 1152, 1197, 1125]
}, index=pd.to_datetime([
    "2010-01-31", "2010-02-28", "2010-03-31", "2010-04-30", "2010-05-31"
]))

labeled = label_business_cycle(macro, spx)

print(labeled)
