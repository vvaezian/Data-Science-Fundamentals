Connecting and getting the webpage as text, and converting to soup object:
```python
from bs4 import BeautifulSoup

res = requests.get(link)
soup = BeautifulSoup(res.text, 'html5lib')
```

`soup.<tag>` retrieves the first `<tag>`. E.G. `soup.p` retrieves the first `<p>...</p>`.
```python
soup.title # <title> ... </title>
soup.title.string # myPageTitle
soup.title.parent # <head> ... </head>
soup.title.parent.name # head
```
- `soup.findall('p')` returns a list containing all `<p>...</p>` tags at every level. The option `limit=n` force it to stop after n items found.
- `soup.find('p')` returns the first `<p>...</p>`. (is it the same as `soup.p` ?)
- `.get([attr])` returns the attribute. E.G `soup.p.get('id')` returns the id of the first `p` tag.
