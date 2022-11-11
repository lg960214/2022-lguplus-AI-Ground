## Structure

```bash
├── notebooks
│
├── data
│   ├── raw_data
│   └── recbole_data
│   
├── src
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
│
├── requirements.txt
└── run.sh
``` 

# git usage

- pr 날리는법

```bash
git checkout -b [issue name]
git add .
git commit -m 'message'
git push -u origin [issue name]
```


- pr 반영하는 법

case 1
```bash
git chekcout master
git merge --no-f [issue-name]
git push -u origin master
```

case 2
github 홈페이지에서 pull request 작성 후, merge 요청

# pr 리뷰 후, 수정하는 법
```bash
git add files
git commit --amend 
git push -f origin [issue-name]
```