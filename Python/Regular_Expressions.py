# Cheetsheet: https://www.debuggex.com/cheatsheet/regex/python

# cappital letters negate. For example \W is negation of \w
# \w = [A-Za-z0-9_]
# . = Any character except newline

import re
msg = "hi, this is a test"
re.findall('\w*', msg)  # ['hi', '', '', 'this', '', 'is', '', 'a', '', 'test', '']
re.findall('\w+', msg)  # ['hi', 'this', 'is', 'a', 'test'] 
re.findall('.*', msg)  # ['hi, this is a test', '']
re.findall('.+', msg)  # ['hi, this is a test']
re.findall('.*s', msg)  # ['hi, this is a tes']
re.findall('.*?s', msg)  # ['hi, this', ' is', ' a tes']  non-greedy search

# Return the first match. If it doesn't match anything, returns None. So it can be used in conditionals
re.match('.*"(.*),(.*).*"', row)  
# refer to matched groups using \g<n> (\n also works but is ambiguous when number is in the text) 
re.sub('(.*)abc(.*)"', '"\g<1>\g<2>"', row)  
try:
    extracted_pattern = re.search('AAA(.+?)ZZZ', text).group(1)  # equivalent of perl -pe 's/AAA(.+?)ZZZ/\1/' test
except AttributeError:
    extracted_pattern = ''
    
# Split my_str on spaces
re.split(r"\s+", my_str)

# Matching the part that starts with an space followed by / (without including the space in the result) until a space (without including the space):
re.findall('(?<=\s)/[^\s]*', log_msg)[0] # ?<= is called 'lookahead'

# difference between match and search functions
re.match('ab', 'abcde')  # <re.Match object; span=(0, 2), match='ab'>
re.search('ab', 'abcde')  # <re.Match object; span=(0, 2), match='ab'>
re.match('cd', 'abcde')  # None
re.search('cd', 'abcde')  # <re.Match object; span=(2, 4), match='cd'>

match = re.search("test", my_sentence)
print(match.start(), match.end())
