
##Description
This data is related to direct marketing campaigns of a Portuguese banking institution. The product they were trying to sell their customers is a term deposit. The marketing campaign is based on phone calls, and often more than one call to the same client is required. 



##Explanation of each column

#### bank client data

1 - age (numeric)

2 - job : type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')

3 - marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown'; note: 'divorced' means divorced or widowed)

4 - education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')

5 - default: has credit in default? (categorical: 'no', 'yes', 'unknown')

6 - balance: client's balance in his accounts (numeric)

7 - housing: has housing loan? (categorical: 'no', 'yes', 'unknown')

8 - loan: has personal loan? (categorical: 'no', 'yes', 'unknown')

#### related with the last contact of the current campaign

9 - contact: contact communication type (categorical: 'cellular', 'telephone') 

10 - day: last contact day of month (numeric)

11 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')

12 - duration: last contact duration, in seconds (numeric)

#### other attributes

13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; -1 means client was not previously contacted)

15 - previous: number of contacts performed before this campaign and for this client (numeric)

16 - poutcome: outcome of the previous marketing campaign (categorical: 'failure', 'nonexistent', 'success')

#### marketing campaign success

17 - y - has the client subscribed a term deposit? (binary: 'yes', 'no')


## Source

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
