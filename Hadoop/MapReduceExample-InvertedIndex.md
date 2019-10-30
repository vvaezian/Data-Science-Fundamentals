https://classroom.udacity.com/courses/ud617/lessons/713848763/concepts/7111190830923

Mapper
````Python
import re
import sys
import csv

def mapper():
  reader = csv.reader(sys.stdin, delimiter='\t')
  
  for line in reader:
    try:
      node_id = line[0].strip('\"')
      node_body = line[4]
      words = [word.strip() for word in re.split('\W+', node_body)]
    except ValueError:
      continue
    for word in words:
      print(word.lower(), node_id)

if __name__ == "__main__":
  mapper()
````
Reducer
````Python
import sys

def reducer():
  prevKey = None
  values = []

  for line in sys.stdin:
    
    data = line.split()
    if len(data) != 2:
      continue
    
    curKey, curValue = data
    
    if prevKey and curKey != prevKey:
      print(prevKey, values, len(values))
      prevKey = curKey
      values = []
      
    values.append(curValue)
    prevKey = curKey

  if prevKey:
    print(prevKey, values, len(values))

if __name__ == "__main__":
  reducer()
````
