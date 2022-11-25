export URL_MODEL="GaussianNB.pkl"

if [[ -f $URL_MODEL ]]
then
  echo "Model also exists"
else
  wget -O $URL_MODEL 'https://drive.google.com/uc?export=download&id=1PIUpjMheDYLKR4vQ2ITwR0xHg5N3e_yw'
  echo "Model was added --" $URL_MODEL
fi


uvicorn main:app --reload --host 127.0.0.1 --port 8000