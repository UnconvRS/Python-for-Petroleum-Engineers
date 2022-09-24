#Modify the y-axis limits so that the maximum Cum. production gets displayed for each well
#
##Write code that will query the database and plot all gas wells
#dcaDF = pd.read_sql_query("SELECT wellID FROM DCAparams WHERE fluid='gas';", conn)
# 
##Write code that will query the database for the well with exponential decline. Print the corresponding wellID
#dcaDF = pd.read_sql_query("SELECT wellID FROM DCAparams WHERE b=0;", conn) 
#Can you stack plot all oil wells and stack plot all gas wells in the field?


import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

import pandas as pd
import sqlite3

#create a database named "DCA.db" in the folder where this code is located
conn = sqlite3.connect("DCA.db")  #It will only connect to the DB if it already exists

#create data table to store summary info about each case/well
cur = conn.cursor()

#Custom Plot parameters
titleFontSize = 18
axisLabelFontSize = 15
axisNumFontSize = 13


#RUN THIS TO CREATE A NEW TABLE
cur.execute("CREATE TABLE DCAparams (wellID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, qi REAL, Di REAL, b REAL, fluid TEXT)")
conn.commit()

dfLength = 24
gasWellID = np.random.randint(1,17,5)   #arguments are low, high, size

for wellID in range(1,18):
    # Load spreadsheet
    fileName = 'DCAwells_Solved/DCA_Well ' + str(wellID) + '.xlsx'
    
    xl = pd.ExcelFile(fileName)
    
    # Load a sheet into a DataFrame by name: df1
    df1 = xl.parse('DCARegression')
    
    rateDF = pd.DataFrame({'wellID':wellID*np.ones(dfLength,dtype=int), 'time':range(1,dfLength+1),'rate':df1.iloc[8:32,1].values})
    rateDF['Cum'] = rateDF['rate'].cumsum()
    
    #insert data into the summary table
    qi = df1.iloc[2,3]
    Di = df1.iloc[3,3]
    b  = df1.iloc[4,3]
    
    
    if wellID in gasWellID:
        cur.execute(f"INSERT INTO DCAparams VALUES ({wellID}, {qi}, {Di}, {b},'gas')")
    else:
        cur.execute(f"INSERT INTO DCAparams VALUES ({wellID}, {qi}, {Di}, {b},'oil')")

    conn.commit()
    
    t = np.arange(1,dfLength+1)
    Di = Di/12   #convert to monthly
    
    if b>0:
        q = 30.4375*qi/((1 + b*Di*t)**(1/b))
        Np = 30.4375*(qi/(Di*(1-b)))*(1-(1/(1+(b*Di*t))**((1-b)/b))) #30.4375 = 365.125/12
    else:
        q = qi*np.exp(-Di*t)
        Np = 30.4375*(qi-q)/Di
        q = 30.4375*q
        
    error_q = rateDF['rate'].values - q
    SSE_q = np.dot(error_q, error_q)
    
    errorNp = rateDF['Cum'].values - Np
    SSE_Np = np.dot(errorNp,errorNp)
    
    
    rateDF['q_model'] = q
    rateDF['Cum_model'] = Np
    # Use DataFrame's to_sql() function to put the dataframe into a database table called "Rates"
    rateDF.to_sql("Rates", conn, if_exists="append", index = False)


    # Read from Rates database table using the SQL SELECT statement
    prodDF = pd.read_sql_query(f"SELECT time,Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)    
    dcaDF = pd.read_sql_query("SELECT * FROM DCAparams;", conn) #this will grab everything in DCAparams table  
    

    #remaining code in loop plots the graphs
    currFig = plt.figure(figsize=(7,5), dpi=100)
    
    # Add set of axes to figure
    axes = currFig.add_axes([0.15, 0.15, 0.7, 0.7])# left, bottom, width, height (range 0 to 1)
    
    # Plot on that set of axes
    axes.plot(prodDF['time'], prodDF['Cum']/1000, color="red", ls='None', marker='o', markersize=5,label = 'well '+str(wellID) )
    axes.plot(prodDF['time'], prodDF['Cum_model']/1000, color="red", lw=3, ls='-',label = 'well '+str(wellID) )
    axes.legend(loc=4)
    axes.set_title('Cumulative Production vs Time', fontsize=titleFontSize, fontweight='bold')
    axes.set_xlabel('Time, Months', fontsize=axisLabelFontSize, fontweight='bold') # Notice the use of set_ to begin methods
    axes.set_ylabel('Cumulative Production, Mbbls', fontsize=axisLabelFontSize, fontweight='bold')
    axes.set_ylim([0, 1200])
    axes.set_xlim([0, 25])
    xticks = range(0,30,5) #np.linspace(0,4000,5)
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticks, fontsize=axisNumFontSize); 
    
    yticks = [0, 400, 800, 1200]
    axes.set_yticks(yticks)
    axes.set_yticklabels(yticks, fontsize=axisNumFontSize); 
    
    #currFig.savefig('well'+str(wellID)+'_Gp.png', dpi=600)




#Syntax to create a foreign key (in SQLite) in the Rates table
cur.execute("ALTER TABLE Rates RENAME TO _old_Rates;")
cur.execute("CREATE TABLE Rates                                  \
(                                                                \
  rateID INTEGER PRIMARY KEY AUTOINCREMENT,                      \
  wellID INTEGER NOT NULL,                                       \
  time INTEGER NOT NULL,                                         \
  rate REAL, \
  Cum REAL, \
  q_model REAL, \
  Cum_model REAL, \
  CONSTRAINT fk_DCAparams                                        \
    FOREIGN KEY (wellID)                                         \
    REFERENCES DCAparams (wellID)                                \
);")

cur.execute("INSERT INTO Rates (wellID, time, rate, Cum, q_model, Cum_model) \
            SELECT wellID, time, rate, Cum, q_model, Cum_model FROM _old_Rates;")
conn.commit()



conn.close()




#stacked plot
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

import pandas as pd
import sqlite3

#create a database named "DCA.db" in the folder where this code is located
conn = sqlite3.connect("DCA.db")  #It will only connect to the DB if it already exists
#create data table to store summary info about each case/well
cur = conn.cursor()

wellID = 1
df1 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 2
df2 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 3
df3 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn) 

labels = ["well 1", "well 2", "well 3"]
fig, ax = plt.subplots()
ax.stackplot(df1['time'], df1['Cum']/1000, df2['Cum']/1000, df3['Cum']/1000, labels=labels)
ax.legend(loc='upper left')
plt.show()



#stacked bar graph
N = 12
ind = np.arange(1,N+1)    # the x locations for the groups
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
width = 0.5       # the width of the bars

p1 = plt.bar(df1['time'][0:N], df1['Cum'][0:N]/1000, width)
p2 = plt.bar(df1['time'][0:N], df2['Cum'][0:N]/1000, width, bottom=df1['Cum'][0:N]/1000)
p3 = plt.bar(df1['time'][0:N], df3['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df2['Cum'][0:N])/1000)

plt.ylabel('Oil Production, Mbbls')
plt.title('Cumulative Production Forecast')
plt.xticks(ind, months, fontweight='bold')
plt.legend((p1[0], p2[0], p3[0]), ('well 1', 'well 2', 'well 3'))

plt.show()



# Primary and Secondary Y-axes
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df1['time'], df1['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df1['time'], df1['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')

plt.show()





#two approaches to load a LAS input file
data1 = np.loadtxt("volve_logs/15_9-F-1B_INPUT.LAS", skiprows=69)
data1DF = pd.read_csv("volve_logs/15_9-F-1B_INPUT.LAS",skiprows=69, sep = '\s+' )

#Load and prepare the data
data = np.loadtxt("WLC_PETRO_COMPUTED_INPUT_1.DLIS.0.las", skiprows=48)
DZ,rho=data[:,0], data[:,1]

#clean data where negative density
DZ=DZ[np.where(rho>0)]
rho=rho[np.where(rho>0)]

print('Investigated Depth',[min(DZ),max(DZ)])

fig = plt.figure(figsize=(5,15), dpi=100)
plt.plot(rho,DZ, color='blue')
plt.xlabel('Density, g/cc', fontsize = 14, fontweight='bold')
plt.ylabel('Depth, m', fontsize = 14, fontweight='bold')
plt.gca().invert_yaxis()  # This is what inverts the direction of the y-axis
plt.show()

titleFontSize = 22
fontSize = 20
#Plotting multiple well log tracks on one graph
fig = plt.figure(figsize=(36,20),dpi=100)
fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1, 6, 1)
plt.grid(axis='both')
plt.plot(rho,DZ, color='red')
plt.plot(rho*1.1,DZ, color='blue')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 2)
plt.grid(axis='both')
plt.plot(rho,DZ, color='green')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 3)
plt.grid(axis='both')
plt.plot(rho,DZ, color='blue')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 4)
plt.grid(axis='both')
plt.plot(rho,DZ, color='black')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 5)
plt.grid(axis='both')
plt.plot(rho,DZ, color='brown')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 6)
plt.grid(axis='both')
plt.plot(rho,DZ, color='grey')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

fig.savefig('well_1_log.png', dpi=600)





##Syntax to add new columns to a table
#cur.execute("ALTER TABLE Rates ADD rateID INTEGER;")
#conn.commit()

#Syntax to delete a table
#cur.execute("DROP TABLE DCAparams;")
#cur.execute("DROP TABLE Rates;")
#conn.commit()
