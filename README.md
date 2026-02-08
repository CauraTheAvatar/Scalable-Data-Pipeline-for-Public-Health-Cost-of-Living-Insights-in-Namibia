# Scalable Data Pipeline for Cost of Living Analysis in Namibia

## Problem Statement
How has inflation evolved in Namibia over time, and how does it coincide with trends in key macro-economic indicators such as GDP per capita and government expenditure?

This project builds a scalable data pipeline to analysee inflation trends in Namibia and their association with macroeconomi-c indicators such as GDP per capita and government expenditure using national-level time series data.

## Scope
The analysis focuses on macroeconomic time series patterns and does not attempt to infer causality or micro-level economic impacts from 1993 to 2024, with projections till 2030.


## How to activate the Virtual Environment
1.  Select the python Interpreter:
  Press Shift + Control + P to open the Command Palette and click on the Python: Select Interpreter.
  Select the Python interpreter that came with your environment (the recommended one).

2.  Use the following command in the terminal:
  .venv\Scripts\activate.ps1

## How to deactivate the virtual environment
In the terminal, type the following:

deactivate


## Libraries useed:
annotated-types==0.7.0
black==26.1.0
blinker==1.9.0
certifi==2026.1.4
cffi==2.0.0
charset-normalizer==3.4.4
click==8.3.1
colorama==0.4.6
contourpy==1.3.3
cryptography==46.0.4
cycler==0.12.1
flake8==7.3.0
Flask==3.1.2
fonttools==4.61.1
greenlet==3.3.1
idna==3.11
isort==7.0.0
itsdangerous==2.2.0
Jinja2==3.1.6
kiwisolver==1.4.9
loguru==0.7.3
MarkupSafe==3.0.3
matplotlib==3.10.8
mccabe==0.7.0
mypy_extensions==1.1.0
numpy==2.4.2
packaging==26.0
pandas==3.0.0
pandera==0.29.0
pathspec==1.0.4
pillow==12.1.0
platformdirs==4.5.1
psycopg2==2.9.11
pycodestyle==2.14.0
pycparser==3.0
pydantic==2.12.5
pydantic_core==2.41.5
pyflakes==3.4.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
python-dotenv==1.2.1
pytokens==0.4.1
requests==2.32.5
seaborn==0.13.2
six==1.17.0
SQLAlchemy==2.0.46
typeguard==4.4.4
typing-inspect==0.9.0
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2025.3
urllib3==2.6.3
Werkzeug==3.1.5
win32_setctime==1.2.0

### Installation commands - Powershell Terminal
pip install flask
pip install psycopg2 pydantic sqlalchemy cryptography
pip install pandas numpy
pip install requests
pip install python-dotenv
pip install loguru
pip install pandera
pip install matplotlib seaborn
pip install black flake8 isort

#### Optional installs
pip install apache-airflow (Day 6 will confirm this choice)

## To check all the installed libraries
pip freeze

# Data Pre-processing
### Identified issues within datasets
IMF data was wide format

### Data melting and cleaning
- Wrote a reusable melt function
- Wrote two separate data loops for the respective source groups
- Normalized to long format
- Added metadata columns
- Dropped missing values
- Converts year to integer
- Value column properly named
- Ensured schema consistency across sources

