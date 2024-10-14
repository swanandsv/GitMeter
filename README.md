GitMeter Django App

The website is a Django based web application. For getting started with Django application please refer tutorial given at: https://www.youtube.com/watch?v=rHux0gMZ3Eg

The back-end of the website can be found at app -> views.py The front-end is based on simple JavaScript, HTML, CSS and can be found under app -> templates

For setting up the website please follow following steps:

Copy or download the Git repo
Please make sure you have python and Django installed for running the application
Other dependencies can be found in ase_project/requirements.txt file. Make sure you have all the required dependies installed on your system before running the application
Once you set up the project please update your GPT API key under app -> views.py
Website also uses Rchilli API for resume parsing. Therefore please make sure you have Rchilli API keys and replace your Rchilli API key under app -> views.py Otherwise comment out the Rchilli API call 
After everything is set up, to start the django project CD into the project and use command 'python manage.py runserver'
