# codenames-bot
The bot to play "Codenames" board game

## Install

```bash
sudo apt-get install python3-tk
git clone https://github.com/vsmelov/codenames-bot
cd codenames-bot
virtualenv -p python3 venv
. venv/bin/activate
pip install -r requirements.txt
wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
```

## Run

```bash
# edit games.py to add yours game
python main.py 
# play around with parameters and datasets
```

## Visualize

```bash
# it's not very helpful to squash 100 dimensions to 3 :-)
python visualize.py
```

## Play

https://play.google.com/store/apps/details?id=com.emeraldpowder.joybox&hl=ru
