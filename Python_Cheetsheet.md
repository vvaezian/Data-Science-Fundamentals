
<em>* Sources: <a href="https://courses.edx.org/courses/course-v1:MITx+6.00.1x_9+2T2016/info" target="_blank">this</a> online course, 
	<a href="https://www.amazon.com/Data-Science-Scratch-Principles-Python/dp/149190142X" target="_blank">this</a> book, and <a href="https://medium.freecodecamp.org/an-a-z-of-useful-python-tricks-b467524ee747"> this </a> post.</em>

## Strings "123"
String Methods
```python
s = 'abcabc'  
s.index('b')  # 1
s.index('z')  # error
s.find('b')  # 1
s.find('z')  # -1
s.count('c')  # 2
s = s.replace('b', 'z')  # "azcazc"
s = s.replace('b', 'z', 1)  # "azcabc"
```
While `index` works for both strings and lists, `find` works only with strings.

Since strings are immutable, there is no method to replace an arbitrary character of a string 
(e.x. replacing the second 'o' in 'foo'). But we can use the following function
(<a href="http://stackoverflow.com/a/9188460/2445273" target="_blank"> here </a>):
```python
def arbit_replace(s, i, c):
    return s[:i] + c + s[i + 1:]

print arbit_replace('foo', 2, 'z')  # foz
```

#### F-Strings (Python 3.6+)
```python
name = 'John'
print(f"name is {name}.")

print(f"{1/3:.3f}")  # 0.333

date=datetime(2018, 1, 1)
print(f"Date is {date:%B %d, %Y}.")  # Date is January 01, 2018.
```

## Lists [1, 2, 3]
```python
list1 = ['a', 'b', 'c']
list1.remove('b')
del list1[-1]
list1.append('z')
list1.insert(1,63)
list1.extend(['d', 'e'])  # to not modify the original list use addition (list1 + ['d', 'e'])
```

#### List Comprehension
```python
list1 = [i for i in range(100) if i not in range(30, 50)]
list2 = [(x, y) for x in range(10)
                for y in range(10)]
list3 = [(x, y) for x in range(10)
                for y in range(x + 1, 10)]
```

## Tuples (1, 2, 3)
Tuples are lists' immutable cousins. 
```python
(3)  # int
(3,)  # tuple
```

## Dictionaries {1:'a', 2:'b', 3:'c'}
```python
aDict={'a':'cat', 'b':'bear', 'c':'cow'}
del aDict['b']
aDict['z']  # KeyError
aDict.get('z')  # None
aDict.get('z', 'nothing')  # nothing
for key, value in aDict.items():
    print "letter: {} Animal: {}".format(key,value)
```
Sort a dictionary by values: 
```python
sorted(myDict.items(), key=lambda item: item[1], reverse=True)
```
	    
#### defaultdict
A defaultdict is like a regular dictionary, except that when you try to look up a key it doesn't contain, it first adds a value for it using a zero-argument function you provided when you created it.

```python
from collections import defaultdict

a = defaultdict(int)
b = defaultdict(str)
c = defaultdict(list)
d = defaultdict(tuple)
e = defaultdict(set)
f = defaultdict(dict)
g = defaultdict(lambda: [5, 8])
g[2][1] = 1

print(a[1])  # 0
print(b[1])  # 
print(c[1])  # []
print(d[1])  # ()
print(e[1])  # set([])
print(f[1])  # {}
print(g[1])  # [5, 8]
print(g.items())  # [(1, [5, 8]), (2, [5, 1])]
```
#### Counter
A `Counter` turns a sequence of values into a defaultdict(int)-like object, mapping keys to counts.
```python
from collections import Counter
c = Counter([0, 1, 2, 0])   # c is (basically) { 0 : 2, 1 : 1, 2 : 1 }
```
A useful method of `Counter` is <samp>most_common</samp>:
```python
# print the 10 most common words and their counts
for word, count in word_counts.most_common(10):
    print(word, count)
```
Different ways of counting occurances of words/letters/symbols in a text given as a list (e.x. by myText.split()):

First approach
```python
word_counts = {}
for word in document:
  if word in word_counts:
      word_counts[word] += 1
  else:
      word_counts[word] = 1
```
Second approach:
```python
word_counts = defaultdict(int)
for word in document:
  word_counts[word] += 1
```
Third approach:
```python
word_counts = Counter(document)
```

## Sets {1, 2, 3}
Set is like list but repetition doesn't matter. {1,1,2} = {1,2}

Adding two sets a and b: <samp>a.union(b)</samp> 

## Exceptions
```python
try:
  ...
except ZeroDivisionError as e:
  ...
except Exception as e: #For all other kinds of exceptions
  ...
```

## Classes
```python
class Coordinate(object):
    def __init__(self, x0, y0):
        self.x = x0
        self.y = y0

c = Coordinate(3,4)
Origin = Coordinate(0,0)
isinstance(c, Coordinate)  # True
```
```python
class Coordinate(object):
    def __init__(self, x0, y0):
        self.x = x0
        self.y = y0
    def __str__(self):
        return "<"+str(self.x)+","+str(self.y)+">"
    def xDifference(self,other):
        return (self.x - other.x)
```
The `__init__()` method (AKA initialization method or class constructor) is called immediately after an instance of the class is created.</br>

In all class methods, `self` refers to the instance whose method was called. But in the specific case of the `__init__()` method, the instance whose method was called is also the newly created object.

Python interpreter translates `obj1 < obj2` into a method call on `obj1` (namely `obj1.__lt__(obj2)`). 
To enable sort operation on instances of a class we should implement the <em>__lt__</em> special method. e.x.:
```python
def __lt__(self, other):
    """return True if self's name is lexicographically
       less than other's name, and False otherwise"""
    if self.lastName == other.lastName:
        return self.name < other.name
    return self.lastName < other.lastName
```
Assume we have a class which defines operations on a set. If we want `len(s)` return the number of members of s, then we need to define <em>len</em> using underscores:
```python
def __len__(self):
    count = 0
    for i in self.vals:
        count += 1
    return count
```
If we don't use underscores, we have to get the length using `s.len()`</li>

### Inheritance
```python
class Person(object):
    def __init__(self, name):
        self.name = name
    def getName(self):
        return self.name

class MITPerson(Person):
    nextIdNum = 0  # next ID number to assign
    def __init__(self, name):
        Person.__init__(self, name)  # initialize Person attributes
        self.idNum = MITPerson.nextIdNum
        MITPerson.nextIdNum += 1
    def getIdNum(self):
        return self.idNum

class UG(MITPerson):
    def __init__(self, name, classYear):
        MITPerson.__init__(self, name)
        self.year = classYear
    def getClass(self):
        return self.year

class Grad(MITPerson):
    pass

def isStudent(obj):
    return isinstance(obj,UG) or isinstance(obj,Grad)
```
As Python has no concept of private variables, leading underscores are used to indicate variables that must not be accessed 
from outside the class. So using local variable names beginning with an underscore is discouraged.
</div>
</details>


## Generators
```python
def genTest():
    yield 1
    yield 2
foo = genTest()
print foo.next()  # 1
print foo.next()  # 2
print foo.next()  # StopIteration Exception
#---------- alternatively
for n in foo:
    print n  # 1 2
```
```python
def genFib():
    fibn_1 = 1 #fib(n-1)
    fibn_2 = 0 #fib(n-2)
    while True:
        next = fibn_1 + fibn_2
        yield next
	fibn_2 = fibn_1
	fibn_1 = next
fib = genFib()
for i in range(5):
    print fib.next()  # 1 2 3 5 8
```
<b>Generator Expressions</b> (alternative of list comprehention to save memory)
The first example above can be written as follows
```python
foo = (i for i in xrange(2))
```

## File Operations
```python
with open('pathToFile', [option]) as f:
  fl = f.readlines()
data = [ eval(line.rstrip('\n')) for line in fl ]
X1, ..., Xn = zip(*data) 
```
[option] can be <samp>'r'</samp> (read) [default], <samp>'w'</samp> (write), <samp>'r+'</samp> (read and write) and <samp>'a'</samp> (append).

`f.readlines()` turns f into a list of strings, each item of list is one line of f (defined by '\n')  
`f.read()` turns f into one string  
`f.readline()` reads the first line as a string

## Copy
<a href="https://stackoverflow.com/a/3975388/2445273">This</a> explains the difference between reference assignment, shallow copying and deep copying.  
`[:]` makes a shallow copy of a string or list.  
`dic.copy()` do the same for dictionaries.  
If our list is a list of lists or list of objects (similar for dictionaries), then we need to do deep copying so that altering the copy doesn't alter the original:
```python
import copy
list2 = copy.deepcopy(list1)
```

## Sorting

```python
l=[(4,'mouse'),(1,'cat'),(3,'horse'),(2,'pog'),(3,'cow'),(2,'dog')]
l.sort(key=lambda x: (-x[0], x[1]))
# [(4, 'mouse'), (3, 'cow'), (3, 'horse'), (2, 'dog'), (2, 'pog'), (1, 'cat')]
```
If an element is a calss instance, we need to use dot notation. E.x.
```python
l.sort(key=lambda x: (x[0].name, x[1].age))
```
	    
## Time
```pythonimport time
from datetime import datetime
time.time()  # returns the value of time in seconds since the Epoch.
datetime.strftime('%Y-%m-%d %H:%M:%S')  # convert datetime to date string
datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")  # convert date string to datetime
myDate.timestamp()  # convert datetime to epoch
datetime.fromtimestamp(t)  # convert epoch to datetime
```
*Runtime*  
```
import time
start = time.time()
```
For small bits of Python code we can use <samp>timeit</samp> (it can be run from <a href="https://docs.python.org/2/library/timeit.html#python-interface">python interface</a> as well): 
`python -m timeit [-n N] [-r N] [statement ...]`
where 
<samp>-n N, (--number=N)</samp> denotes how many times to execute ‘statement’, and <samp>-r N, (--repeat=N)</samp> denotes how many times to repeat the timer (default 3).
	    
	    
## Random Numbers
```python
import random

r_num1 = random.random()  			# random float in [0, 1)
r_num2 = random.randrange(a, b)  		# random integer in [a, b)
r_num3 = random.uniform(a, b)  			# random float in [a, b)
r_num4 = random.normalvariate(mu, sigma)  	# random float from normal dist of mu and sigma

shuffled_list = random.shuffle(my_list)  	# shuffles the list
chosen_member = random.chice(my_list)  		# chooses a member of the list
chosen_members = random.sample(my_list, 4)  	# chooses 4 member of the list without replacement 
chosen_members_with_replacement = [random.choice(my_list) for _ in range(3)]

# use random.seed to get reproducible results
random.seed(5)
print random.random()  # 0.62290169489
random.seed(5)
print random.random()  # 0.62290169489
```

## zip
```python
zip(['a', 'b', 'c'], [1, 2, 3])  # [('a', 1), ('b', 2), ('c', 3)]
zip(('a', 1), ('b', 2), ('c', 3))  # [('a', 'b', 'c'), (1, 2, 3)]
data = [('a', 1), ('b', 2), ('c', 3)]
zip(*data)  # [('a', 'b', 'c'), (1, 2, 3)]
```
   
## `*args` and `**kwargs`
The general use of these is to produce higher-order functions whose inputs can accept arbitrary arguments.
When we define a function using `*args` and `**kwargs`, args is a tuple of its unnamed arguments
and kwargs is a dict of its named arguments:
```python
def test(*args, **kwargs):
    print "unnamed args:", args
    print "keyword args:", kwargs

test(1, 2, key="word", key2="word2")  # unnamed args: (1, 2)
                                      # keyword args: {'key2': 'word2', 'key': 'word'}
```
It works the other way too:
```python
def test2(x, y, z):
    return x + y + z

list123 = [1, 2, 3]
print(test2(*list123))  # 6
dict123 = {'a': 1, 'b': 2, 'c': 3}
print(test2(*dict123))  # 'abc'
print(test2(**dict123))	# 6
```

## Web Scraping
```python
import requests
from bs4 import BeautifulSoup

res = requests.get(url)
soup = BeautifulSoup(res.text)
print(soup.prettify())
divs = soup.find_all('div', {'class':'test'})  # for multiple classes use a list ['test1', 'test2']
divs[3].get('href')
```
The package <a href="https://newspaper.readthedocs.io/en/latest/" > newspaper </a> may be helpful for some use-cases.
	
	
## Misc
- <b>Difference between `==` and `is`</b>
    The equality operator (`x == y`) tests the values of x and y for equality, while the identity operators (`is`) tests two objects to see whether they refer to the same object in memory. 
    In general, it may be the case that `x == y`, but `x is not y`.
```python
a = [1, 2, 3]
b = a
b is a  # True
b == a  # True

c = a[:]
c is a  # False
c == a  # True
```
- `dir` lists attributes of an object. E.g. <sampl>dit("hi")</samp> lists all strng methods.
- `inspect` is handy for looking at source code modules:

```python
import inspect
import queue	# Python 3
print(inspect.getsource(queue))
```
- `sh` module makes it easy to work with standard os and subprocess libraries:
```python
import sh
sh.pwd()
sh.mkdir('newDir')
sh.touch('newFile.txt')
sh.whoami()
sh.echo('This is cool!')
```
`pythonpy` is the opposite. Uses python is command-line.

- Command-line arguments
`sys.argv` is the list of command-line arguments. So <samp>sys.argv[0]</samp> is the name of the program itself.

- `tqdm`: `from tqdm import tqdm` (for Notebooks: `from tqdm.notebook import tqdm`
