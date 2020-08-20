#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:27:39 2020

@author: muhammed
"""
#SAYILAR VE STRINGLERE GIRIŞ

#sayısal değişkenler
9 #kesikli integer
9.2 #sürekli float
9*9.2

print("HELLO WORLD")

type(9)
type(9.2)
type("HELLO WORLD") # string
type(1+2j)

#STRINGLERE YAKINDAN BAKALIM  operations
123
type(123)
str(123)+"123"
"a" + "b"
'a' 'b'
"a"+"-b"
#"a"-"a" TypeError hatası alınabilir.
"a"*3

#STRING METOTLARI

#LEN()

variable = "geleceği_yazanlar"
len(variable)
print(variable.upper())
print(variable.lower())
print(variable.isupper())
if(variable.islower()):
    variable = variable.upper()
print(variable)
a=10
b=9
a*b
del variable
del a 
del b


#REPLACE 
x = "geleceği_yazanlar "
x.replace("e", "a",) 

#STRIP Ne verirsen onu atar string içinden
x.strip() 

#CAPITALIZE baştaki harfi büyük yapar
x.capitalize()

#SUBSTRING
x[0:-2:2] # 0.index den -2.index e 2 şer arta arta , *2.index dahil değil
#x[20] #IndexError hatası alınabilir.
#YARDIMCI METOTLAR dir(...)

dir(str)
dir(int)


#TYPE DÖNÜŞÜMLERİ

first = input() #10
second = input() #20
first + second  #1020
int(first)+int(second) #30


#PRINT FONKSİYONU

print("HELLO AI ERA")

print("geleceği","yazanlar",end=",")
print("geleceği","yazanlar",sep="|")
print() # Tümünü seçip ctrl+ı yapınca help kısmında fonksiyon açıklanıyor


#"10"+2
"_Python_".strip("_")


#VERİ YAPILARI

#Listeler

#[]
#list()

notlar= [90,80,70,50]
type(notlar)

liste = ["a",19.3,90,notlar]
type(liste[0])
for i in liste:
    print(i)
print(i)
del liste
del notlar
del i

liste = [10,20,30,40,50]

liste[1]
#liste[6]  #IndexError list index out of range
liste[0:2]
liste[0::2]
liste[2:len(liste)-1]
liste[2:-1]

del liste

liste = ["ali","veli","berkcan","ayse"]
liste
liste[1] = "velinin babasi"
liste
liste[0:3] = "alinin babasi","velinin babası","berkcanın babası"
liste += ["kemal"]
liste
liste.remove("kemal")
liste
liste.append("kemal")
liste 

dir(liste)

liste.append("berkacan")
liste.remove("berkacan")
liste[0] = "ayse"
liste.remove("ayse")
liste


#insert
liste.insert(0,"ayse")
liste
liste.insert(10,"deneme")
liste


#pop
liste.pop()
liste.pop(0)
liste.count("deneme")


#copy
yedek = liste.copy()

liste2 = ["asd"]
liste.extend(liste2)
liste
liste.append(liste2)
liste
liste.index("asd")
liste.index("deneme")
liste.reverse()
liste

#yedek.sort() # TypeError not supported between instances of str and list
liste2 = [10,30,20]
liste2.sort()

liste2.clear()
del liste2

del liste
del yedek


# TUPLE Değiştirilemez. Sıralıdır.

t1 = ("ali","veli",1,2,3,[1,2,3,4])
t2 = "ali","veli"
t3 = ("eleman")
type(t3) # str basar
t3 = ("eleman",)
type(t3) # tuple basar
del t2,t3


#t1[1] = "velinin babası" # TypeError not support assignment
t1[5].append(5)
t1
del t1
# DICTIONARY 

sozluk = {"REG" : "Regression Modeli", "LOJ":"Lojistik Regression","CART":"Classification and Reg"}

len(sozluk)
#sozluk[0] # KeyError 0 not found
sozluk["REG"]

for i in sozluk.keys():
    print(sozluk.get(i))
for i in sozluk.items():
    print(i[0],i[1],sep=":")
    
del i

sozluk["GBM"] = "Gradient Boosting Mac"
sozluk
sozluk["REG"] = "Coklu Doğrusal Regresyon"
sozluk
type(sozluk.values())
for i in sozluk.values():
    print(i)
t = ("eleman",)
sozluk[t] = "deneme123"
sozluk


# Setler Sırasızdır.Eşsizdir.Unique değerler taşır. Kümelere benzer.

s = set("deneme")
s = {1,"a","ali",123}
s = set(t)

l = ["ali","lütfen","ata","bakma","uzaya","git","git","ali","git"]
s = set(l)
len(s) # unique eleman sayısı
name = "muhammed"
len(set(name)) #muhammed içindeki unique character sayısı

s[0] # TypeError is not subscriptable

del i,l,name,s,sozluk,t

#Eleman ekleme çıkarma

l = ["geleceği","yazanlar"]
s = set(l)
dir(s)
s.add("ile")
s
s.add("ile")
s.remove("ile")
s.remove("ile")
s.discard("ile") # Hata mesajı göstermeden ,varsa silme

del s,l
#Küme işlemleri

#difference

set1 = set([1,3,5])
set2 = set([1,2,3])

set1.difference(set2)
set2.difference(set1)
set1 - set2
set2 - set1
set1.symmetric_difference(set2)

#intersection

set1.intersection(set2)
set2.intersection(set1)
set1 & set2
set2 & set1

#union

birlesim = set1.union(set2)
set1.intersection_update(set2)
set1

del birlesim
# Setlerde Sorgu İşlemler
set1 = set([7,8,9])
set2 = set([5,6,7,8,9,10])

#kesisim olduğunu sorgulama

set1.isdisjoint(set2) # İki kümenin kesişimi boş mu


#alt küme mi ve kapsıyor mu

set1.issubset(set2)
set2.issubset(set1)
set2.issuperset(set1)



#FONKSIYONLAR

print
?print

3**3
25%2
25/2

def kare_al(x):
    return x**2
print(kare_al(4))

def square(x):
    print("Girilen sayı  {}, Karesi {} ".format(str(x), str(x**2)))

square(4)

def carpma_yap(x,y=1):
    print(x*y)
carpma_yap(3)    
carpma_yap(x=3)
carpma_yap(y=2,x=3)

del carpma_yap,kare_al,set1,set2,square

#Local ve Global Değişkenler

x = 10
y = 20 # global

def carpma_yap(x,y): # x ve y local değişkenler. Bir scope içerisindeler.
    return x*y

print(carpma_yap(2, 3))

#KARAR KONTROL YAPILARI if elif else
x = 10

x == 10 # True == != >= <= 

if x < 5:
    print("asd")
elif x < 10:
    print("qwe")
else:
    print("zxc")
    
del x
# FOR Döngüsü

people = ["ali","veli"]
for i in people:
    print(i)
for i in range(len(people)):
    print(people[i])
    

#NESNE YÖNELİMLİ PROGRAMLAMA

class VeriBilimci():
    print("Bu bir sınıftır")
    
del VeriBilimci
#Class attributes

class VeriBilimci():
    bolum = ""
    sql = "Evet"
    deneyim_yili = 0
    bildigi_diller = []
VeriBilimci.sql   
VeriBilimci.sql = "Hayır"
VeriBilimci.sql

#Sınıf Örneklendirmesi

ali = VeriBilimci()
ali.sql
ali.deneyim_yili
ali.bolum
ali.bildigi_diller.append("Python")
ali.bildigi_diller

veli = VeriBilimci()
veli.bildigi_diller # Python veli yede geldi.Yanlış bir durum.
del VeriBilimci

class VeriBilimci():
    calisanlar = []
    def __init__(self,name=""):
        self.name = name
        self.bildigi_diller = []
    def learn_language(self,x):
        self.bildigi_diller.append(x)
    
    
ali = VeriBilimci("ali")
VeriBilimci.calisanlar.append(ali)
ali.bildigi_diller.append("Python")
ali.bildigi_diller

veli = VeriBilimci("veli")
VeriBilimci.calisanlar.append(veli)
veli.bildigi_diller.append("R")
veli.bildigi_diller

for calisan in VeriBilimci.calisanlar:
    print(calisan.name,calisan.bildigi_diller)
    
ali.learn_language("Sql")
ali.bildigi_diller

#Inheritance

class People():
    def __init__(self,name=""):
        self.name=name
class Employee(People):
    def __init__(self,salary):
        self.salary=salary

ali = Employee(1500)
ali.name

#Fonksiyonel Programlama

#Stateless Function 

A = 5
def impure_sum(b):
    return b + A
def pure_sum(a,b):
    return a + b

#Anonymous Function

new_sum = lambda a,b : a+b
new_sum(4, 5)


sırasız_liste = [("b",3),("a",8),("d",12),("c",1)]
sırasız_liste

sorted(sırasız_liste,key=lambda x:x[1])


#Vektörel Operasyonlar

#OOP
a = [1,2,3,4]
b = [2,3,4,5]

ab = []

for i in range(len(a)):
    ab.append(a[i]*b[i])
ab

#FP
import numpy as np
a = np.array([1,2,3,4])
b = np.array([2,3,4,5])
a*b

#map filter reduce

liste =[1,2,3,4,5]

for i in range(len(liste)):
    liste[i] += 10
liste

#map
list(map(lambda x:x*10,liste))

#filter

liste = [1,2,3,4,5,6,7,8,9,10]

list(filter(lambda x: x%2 == 0,liste))

#reduce

from functools import reduce

liste =[1,2,3,4]

reduce(lambda a,b: a+b , liste)

del A,Employee,People,VeriBilimci,a,ab,ali,b,calisan
del i,impure_sum,liste,new_sum,pure_sum,reduce,sırasız_liste,veli


#Modül Oluşturmak
import Hesap as he
from Hesap import yeni_maas

he.yeni_maas(1000)
yeni_maas(2000)


# Try Except 

a = 10
b = "2"
try:
    try:
        a/b
    except ZeroDivisionError:
        print("zero exception")
        raise Exception
    except TypeError:
        print("str - int - nooooo!")
except:
    print("other exceptions")
finally:
    print("finally part")
    




