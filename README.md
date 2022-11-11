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

# Git Usage

## Basic Rule
```bash
git pull # 원격 저장소 최신 업데이트 내역을 받아옴
git fetch # 원격 저장소의 브랜치 및 정보들에 대한 업데이트
```

## Make New Branch
<img width="379" alt="image" src="https://user-images.githubusercontent.com/47301926/201261568-ca666504-14a7-4abc-a421-c94a428142d3.png">

일반적으로 main 브랜치 위에서 새로운 branch를 생성한 후,
`git fetch`를 하게 되면, 터미널 및 Vscode에서 생성한 브랜치가 보이게 됨
(git branch로 확인 가능)



## How to Pull Request

```bash
git checkout -b [issue name]
git add .
git commit -m 'message'
git push -u origin [issue name]
```


## PR Merge Case

### CLI
```bash
git chekcout main
git merge --no-f [issue-name]
git push -u origin
```

case 2
github 홈페이지에서 pull request 작성 후, merge 요청

# pr 리뷰 후, 수정하는 법
```bash
git add files
git commit --amend 
git push -f origin [issue-name]
```
