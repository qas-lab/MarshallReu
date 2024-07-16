# This file will work with creating triaging for developers. 

''' 
Class for creating a developer and placing the devloper into a list
'''
class Developers:

    def __init__(self, name, amount, type):

        self.name = name
        self.amount = amount
        self.type = type
        self.bugList = dict()

    def addToDict(self, amount, type):

        if type not in self.bugList.keys():

            self.bugList[type] = amount
        
        return 
    
    def getName(self):
        return self.name
    
    def getType(self):
        return self.type
    
    def getAmount(self):
        return self.amount
    
    def topBugCategory(self):

        

        return 
    
'''
Triage class for developers
'''
class Triagers:

    def __init__(self, devs):

        self.devs = devs
        self.list = dict()
    
    def addToList(list, type, devs):

        list[type] = devs
        
        return list

testDev1 = Developers(name='Mike', amount=250, type='Debug')
testDev1.addToDict(250, 'Debug')
testDev1.addToDict(20, 'Core')

print(testDev1.bugList)
