Summarry of what I learn in this fiels.

**The material is for my personal use and I don't take credit for the material.*


### Comparison of Python
```sql
select top 10 id
from tags
order by count desc

select day(creationdate), count(*)
from posts 
where tags like '%javascript%'
and creationdate > '2019-11-01 00:00:00'
group by day(creationdate)
```
