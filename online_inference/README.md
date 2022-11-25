# Технопарк, МГТУ, ML-21, Кежаев Максим

MLOps Homework 2

Start:
```
cd online_inference/
```

Build docker image:

```
docker build -t hw2-kezhaev:1.0.0 .
```

Pull docker image:

```
docker pull makezh/hw2-kezhaev:1.0.0
```

Quick run:

```
docker run -p 8000:8000 hw2-kezhaev:1.0.0 
```

Local run:
```
/bin/zsh run.sh
```

Service is running on _http://localhost:8000


### Docker optimization

1. Using python:3.8-slim
2. pip install with --no-cache 
3. Copying only necessary files to docker image
4. Loading model from Google Drive instead of building from hw1

Size: 509 MB
