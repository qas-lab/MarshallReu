# This file will work with creating triaging for developers. 

''' 
Class for creating a developer and placing the devloper into a list
'''
class Developers:
    def __init__(self, name, amount):
        self.name = name
        self.amount = amount
        self.type = list()
        self.bugList = dict()

    def addToDict(self, amount, type):
        if type not in self.bugList:
            self.bugList[type] = amount
        else:
            self.bugList[type] += amount
        
        if type not in self.type:
            self.type.append(type)

    def getName(self):
        return self.name
    
    def getType(self):
        return self.type
    
    def getAmount(self):
        return self.amount
    
    def topBugCategory(self):
        if not self.bugList:
            return None
        return max(self.bugList, key=self.bugList.get)

    def getBugCount(self, type):
        return self.bugList.get(type, 0)

    def getTotalBugs(self):
        return sum(self.bugList.values())

    def __str__(self):
        return f"Developer: {self.name}, Total Bugs: {self.getTotalBugs()}"
    
'''
Triage class for developers, I am unsure how I will need to use this file
'''
class Triagers:

    def __init__(self, devs):

        self.devs = devs
        self.list = dict()
    
    def addToList(list, type, devs):

        list[type] = devs
        
        return list


