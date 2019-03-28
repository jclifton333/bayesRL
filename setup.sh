sudo apt-get update
sudo apt-get install git
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
git clone https://github.com/jclifton333/bayesRL
cd bayesRL
python3 -m pip install -r requirements.txt --user

