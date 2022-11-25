export MODEL_NAME="GaussianNB.pkl"

if [[ -f $MODEL_NAME ]]
then
  echo "Model also exists"
else
  wget -O $MODEL_NAME 'https://drive.google.com/uc?export=download&id=1PIUpjMheDYLKR4vQ2ITwR0xHg5N3e_yw'
  echo "Model was added --" $MODEL_NAME
fi


uvicorn main:app --reload --host 127.0.0.1 --port 8000