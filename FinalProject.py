import pandas
import pandas as pd
from sklearn import linear_model

df=pd.read_csv(r"D:\ALL PROJECT\SLD New Project\Final.csv")

num = int(input("How many data do you want as Training data in percentage(not including % symbol):"))
num = num/100

l1= df['Do you take medicine for Lever'].tolist()
#print("length = ",len(x1_1))
length = len(l1)
train = num*length
train = int(train)
#print('train= ',train)

l2=df['Age'].tolist()
l3=df['Mode of Transport'].tolist()
l4=df['How time you socialize through Social Media(e.g.facebook, twitter)?'].tolist()
l5=df['Exercising at a gym/free hand regularly?'].tolist()
l6=df['Stops smoking recently?'].tolist()
l7=df['Do you like to take salt additionally in food you eat?'].tolist()
l8=df['Any family history of coronary heart disease?'].tolist()
l9=df['Do you think you need to reduce some weight?'].tolist()
l10=df['Are you trying to give up taking junk food'].tolist()
l11=df['Do you take medicine for Kidney?'].tolist()
l12=df['Are you trying to give up taking meat?'].tolist()
l13=df['Do you frequently take alcohol?'].tolist()

t1=df['Have high blood pressure/high cholestrol/both'].tolist()

x = df[['Do you take medicine for Lever','Age','Mode of Transport','How time you socialize through Social Media(e.g.facebook, twitter)?','Exercising at a gym/free hand regularly?','Stops smoking recently?','Do you like to take salt additionally in food you eat?','Any family history of coronary heart disease?','Do you think you need to reduce some weight?','Are you trying to give up taking junk food','Do you take medicine for Kidney?','Are you trying to give up taking meat?','Do you frequently take alcohol?']][0:train]    

#print("length= ",len(x))
#x = [x1,x2,x3,x4]
#print("Type(x) = ",type(x))
y = df['Have high blood pressure/high cholestrol/both'][0:train]
regr = linear_model.LinearRegression()

regr.fit(x.values,y)

#print("type= ",type(y))
#print("Type(y1) = ",type(y1))

y1 = df['Have high blood pressure/high cholestrol/both'][train+1:length].tolist()

tp = 0
tn = 0
fp = 0
fn = 0
#print("Original Value                           Predicted Value")
list1=[]
temp1=[]
for i in range(train+1,length):
    predictedValue = regr.predict([[l1[i],l2[i],l3[i],l4[i],l5[i],l6[i],l7[i],l8[i],l9[i],l10[i],l11[i],l12[i],l13[i]]])
      
    predictedValue=float(predictedValue)
    
    temp1.append(predictedValue)

    if (predictedValue >= 0.5 and predictedValue < 1.5):
        list1.append(1)
    elif (predictedValue >= 1.5 and predictedValue <2.5):
       list1.append(2)
    elif (predictedValue >= 2.5 and predictedValue <3.5):
       list1.append(3)
    else:
        list1.append(0)

for i in range(len(list1)):
    #print(y1[i],"            ",temp1[i])
    #print(y1[i],"            ",list1[i])
    print(y1[i],"            ",list1[i])
    print(y1[i],"            ",list1[i])
    if((y1[i]==1 and list1[i]==1)or(y1[i]==2 and list1[i]==2)or(y1[i]==3 and list1[i]==3)):
        tp=tp+1
    elif(y1[i]==0 and list1[i]==0):
        tn=tn+1
    elif((y1[i]==0 and list1[i]==1)or(y1[i]==0 and list1[i]==2)or(y1[i]==0 and list1[i]==3)or(y1[i]==1 and list1[i]==2)or(y1[i]==2 and list1[i]==3)or(y1[i]==1 and list1[i]==3)):
        fp=fp+1
    elif((y1[i]==1 and list1[i]==0)or(y1[i]==2 and list1[i]==0)or(y1[i]==3 and list1[i]==0)or(y1[i]==2 and list1[i]==1)or(y1[i]==3 and list1[i]==1)or(y1[i]==3 and list1[i]==2)):
        fn=fn+1
#print("Accurate= ",(tp+tn))
#print("Original= ",len(y1))

print("\n\nConfusion Matrix\n")
print(tp,"    ",fp)
print(fn,"    ",tn)

print("\n\n  Accuracy::", (((tp+tn)/(tp+tn+fp+fn)))*100)
sen1 = ((tp/(tp+fn))*100)                       
print("\nSensitivity ::", sen1)

print("Specitivity :: ", ((tn/(tn+fp))*100))
pre1 = (tp/(fp+tp)*100)
print("PRECISION  ::",pre1)

print("F1 SCORE :: ",(2*(sen1*pre1)/(sen1+pre1))/100)


print("\n:::: 10 Fold Cross Validation ::::\n")

len1 = int(length/10)

z1 =  df[['Do you take medicine for Lever','Age','Mode of Transport','How time you socialize through Social Media(e.g.facebook, twitter)?','Exercising at a gym/free hand regularly?','Stops smoking recently?','Do you like to take salt additionally in food you eat?','Any family history of coronary heart disease?','Do you think you need to reduce some weight?','Are you trying to give up taking junk food','Do you take medicine for Kidney?','Are you trying to give up taking meat?','Do you frequently take alcohol?']][len1+1:length]

y_1 = df['Have high blood pressure/high cholestrol/both'][len1+1:length]

regr_1 = linear_model.LinearRegression()

regr_1.fit(z1.values,y_1)

pre1=[]
for i in range(len1+1):
    p1= regr_1.predict([[l1[i],l2[i],l3[i],l4[i],l5[i],l6[i],l7[i],l8[i],l9[i],l10[i],l11[i],l12[i],l13[i]]])
    #,l13[i],l15[i],l16[i],l17[i],l18[i],l19[i],l20[i]]])
    #,l21[i],l22[i],l23[i]]])
    p1=float(p1)
    #p1=round(p1,1)
    if (p1 >= 0.5):
        pre1.append(1)
    else:
       pre1.append(0)
tp1=0
tn1=0
fp1=0
fn1=0



for i in range(len1):
    #print(t1[i],"            ",pre1[i])
    if((t1[i]==1 and pre1[i]==1)or(t1[i]==2 and pre1[i]==2)or(t1[i]==3 and pre1[i]==3)):
        tp1=tp1+1
    elif(t1[i]==0 and pre1[i]==0):
        tn1=tn1+1
    elif((t1[i]==0 and pre1[i]==1)or(t1[i]==0 and pre1[i]==2)or(t1[i]==0 and pre1[i]==3)or(t1[i]==1 and pre1[i]==2)or(t1[i]==2 and pre1[i]==3)or(t1[i]==1 and pre1[i]==3)):
        fp1=fp1+1
    elif((t1[i]==1 and pre1[i]==0)or(t1[i]==2 and pre1[i]==0)or(t1[i]==3 and pre1[i]==0)or(t1[i]==2 and pre1[i]==1)or(t1[i]==3 and pre1[i]==1)or(t1[i]==3 and pre1[i]==2)):
        fn1=fn1+1

#print("::Confusion Matrix For 1st 10 Fold Data::\n")
#print(tp1,"    ",fp1)
#print(fn1,"    ",tn1)

print("\n\n  Accuracy of 1st 10 fold Data::", (((tp1+tn1)/(tp1+tn1+fp1+fn1)))*100)
#accuracy_of_cross.append((((tp1+tn1)/(tp1+tn1+fp1+fn1)))*100)                   
print("\nSensitivity of 1st 10 fold Data:: ::", ((tp1/(tp1+fn1))*100))
#sen_of_cross(tp1/(tp1+fn1))*100
print("Specitivity of 1st  10 fold Data:: :: ", ((tn1/(tn1+fp1))*100))
#sp_of_cross(tn1/(tn1+fp1))*100
print("PRECISION of 1st  10 fold Data:: ::",(tp1/(fp1+tp1)*100))
#pre_of_cross(tp1/(fp1+tp1)*100)
a=(tp1/(tp1+fn1))
b=tp1/(fp1+tp1)
print("F1 SCORE :: ",(2*a*b)/(a+b))



def function1(n1,up,low,list):
    z2_1 = df[['Do you take medicine for Lever','Age','Mode of Transport','How time you socialize through Social Media(e.g.facebook, twitter)?','Exercising at a gym/free hand regularly?','Stops smoking recently?','Do you like to take salt additionally in food you eat?','Any family history of coronary heart disease?','Do you think you need to reduce some weight?','Are you trying to give up taking junk food','Do you take medicine for Kidney?','Are you trying to give up taking meat?','Do you frequently take alcohol?']][0:up]
    z2_2=df[['Do you take medicine for Lever','Age','Mode of Transport','How time you socialize through Social Media(e.g.facebook, twitter)?','Exercising at a gym/free hand regularly?','Stops smoking recently?','Do you like to take salt additionally in food you eat?','Any family history of coronary heart disease?','Do you think you need to reduce some weight?','Are you trying to give up taking junk food','Do you take medicine for Kidney?','Are you trying to give up taking meat?','Do you frequently take alcohol?']][low:length]
    z2=pd.DataFrame()
    z2=pd.concat([z2_1,z2_2])
    #print(z2)
    y_2_1 = df['Have high blood pressure/high cholestrol/both'][0:up]
    y_2_2=df['Have high blood pressure/high cholestrol/both'][low:length]
    y_2=pd.DataFrame()
    y_2=pd.concat([y_2_1,y_2_2])
    reg = linear_model.LinearRegression()
    reg.fit(z2.values,y_2)
   
    pre2=[]
    for i in range(up,low):
        p2= reg.predict([[l1[i],l2[i],l3[i],l4[i],l5[i],l6[i],l7[i],l8[i],l9[i],l10[i],l11[i],l12[i],l13[i]]])
        p2=float(p2)
        #p2=round(p2,1)
        #print("p2= ",p2)
        if (p2 >= 0.5):
            pre2.append(1)
            #print(pre2)
        else:
            pre2.append(0)
            #print(pre2)
    #print("pre2= ",pre2)
    #print()
    #print(list)
    
    tp1=0
    tn1=0
    fp1=0
    fn1=0
    for i in range(0,len(pre2)):
        #print(list[i],"            ",pre2[i])
        if((list[i]==1 and pre2[i]==1)or(list[i]==2 and pre2[i]==2)or(list[i]==3 and pre2[i]==3)):
            tp1=tp1+1
        elif(list[i]==0 and pre2[i]==0):
            tn1=tn1+1
        elif((list[i]==0 and pre2[i]==1)or(list[i]==0 and pre2[i]==2)or(list[i]==0 and pre2[i]==3)or(list[i]==1 and pre2[i]==2)or(list[i]==2 and pre2[i]==3)or(list[i]==1 and pre2[i]==3)):
            fp1=fp1+1
        elif((list[i]==1 and pre2[i]==0)or(list[i]==2 and pre2[i]==0)or(list[i]==3 and pre2[i]==0)or(list[i]==2 and pre2[i]==1)or(list[i]==3 and pre2[i]==1)or(list[i]==3 and pre2[i]==2)):
            fn1=fn1+1

    #print("\n\n:: Confusion Matrix ::\n")
    #print(tp1,"    ",fp1)
    #print(fn1,"    ",tn1)

    

    print("\n\n  Accuracy of ",n1,"  10 fold Data::", (((tp1+tn1)/(len(pre2))))*100)
     #print("\n\n  Accuracy of ",n1,"th  10 fold Data::", (((tp1+tn1)/(len(pre2))))*100)
    #accuracy_of_cross.append((((tp1+tn1)/(tp1+tn1+fp1+fn1)))*100)                   
    print("\nSensitivity of ",n1,"th 10 fold Data:: ::", ((tp1/(tp1+fn1))*100))
    #sen_of_cross(tp1/(tp1+fn1))*100
    print("Specitivity of ",n1,"th 10 fold Data:: :: ", ((tn1/(tn1+fp1))*100))
    #sp_of_cross(tn1/(tn1+fp1))*100
    print("PRECISION of ",n1,"th 10 fold Data:: ::",(tp1/(fp1+tp1)*100))
    #pre_of_cross(tp1/(fp1+tp1)*100)
    a=(tp1/(tp1+fn1))
    b=tp1/(fp1+tp1)
    print("F1 SCORE :: ",(2*a*b)/(a+b))
   

t2=[]
for i in range(14,27):
    t2.append(t1[i])
function1(2,14,27,t2)

t3=[]
for i in range(27,40):
    t3.append(t1[i])
function1(3,27,40,t3)

t4=[]
for i in range(40,53):
    t4.append(t1[i])
function1(4,40,53,t4)

t5=[]
for i in range(53,66):
    t5.append(t1[i])
function1(5,53,66,t5)

t6=[]
for i in range(66,79):
    t6.append(t1[i])
function1(6,66,79,t6)

t7=[]
for i in range(79,92):
    t7.append(t1[i])
function1(7,79,92,t7)

t8=[]
for i in range(92,105):
    t8.append(t1[i])
function1(8,92,105,t8)

t9=[]
for i in range(105,118):
    t9.append(t1[i])
function1(9,105,118,t9)

t10=[]
for i in range(118,length):
    t10.append(t1[i])
function1(10,118,length,t10)

