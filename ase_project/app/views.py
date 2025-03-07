from django.shortcuts import render,redirect
from .forms import ResumeForm
from PyPDF2 import PdfReader
from openai import OpenAI
import re
import requests
from django.http import JsonResponse
import threading
import logging
from django.contrib.sessions.models import Session
import subprocess
import tempfile
from datetime import datetime
import base64
import json
import concurrent.futures

# Create your views here.

#code_quality_status = {}

base_url = "https://api.github.com"
access_token = "Enter your access token here"
headers = {"Authorization": f"Bearer {access_token}"}

# Create a global lock
lock = threading.Lock()


api_key = "Replace your GPT API Key"
#api_key = openai.api_key
client = OpenAI(api_key=api_key)
matching_codes_count_2 = 0
matching_codes_count = 0
total_forked_repos = 0
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

code_quality_data = {}

non_code_quality_data = {}


def _calculate_non_code_quality_metrics(username):
    # Perform the actual calculation of non-code quality metrics here
    non_code_quality_metrics = gather_non_code_quality_metrics(username)
    non_code_quality_data[username] = non_code_quality_metrics

def _run_code_quality_cal(username):
    averages = calculate_code_quality_metrics(username)
    #logging.debug(f"Calculated averages for {username}: {averages}")
    code_quality_data[username] = {'status': 'completed', 'metrics': averages}
    #logging.debug(f"Updated code_quality_data for {username}: {code_quality_data[username]}")

def calculate_non_code_quality_metrics(username, technologies_from_session):
    non_code_quality_metrics = gather_non_code_quality_metrics(username, technologies_from_session)
    with lock:
        non_code_quality_data[username] = {'status': 'completed', 'metrics': non_code_quality_metrics}

def run_code_quality_cal(username):
    averages = calculate_code_quality_metrics(username)
    with lock:
        code_quality_data[username] = {'status': 'completed', 'metrics': averages}

def count_lines_of_code(code):
    lines = code.split('\n')
    return len(lines)

def is_valid_code_file(filename):
    valid_extensions = ['.py', '.java']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def has_core_logic(filename, content):
    config_related_terms = ['config', 'settings', 'manage', 'models', 'admin', 'urls', '__init__', "package.json", "dockerfile", "nginx.conf", "index.html", "server.js", "app.py",
            "application.properties", "application.yml", "pom.xml", "build.gradle", "web.xml", "applicationcontext.xml",
            "manage.py", "settings.py", "urls.py", "wsgi.py", "gunicorn.conf.py", "supervisord.conf", "pytest.ini", "tox.ini",'Database.java','database.java','ApiResource.java','db.java', 'setup.java','conf.java']
    for term in config_related_terms:
        if term in filename.lower():
            #print("Config related file")
            return False
    return True    


def contains_code_language(username, repo_name, language):
    languages_url = f"https://api.github.com/repos/{username}/{repo_name}/languages"
    languages_response = requests.get(languages_url, headers=headers)

    if languages_response.status_code == 200:
        languages_data = languages_response.json()
        return language in languages_data
    else:
        #print(f"Error: Unable to fetch repository languages. Status Code: {languages_response.status_code}")
        return False



def get_code_from_repository(username):
    global matching_codes_count_2
    global matching_codes_count
    global total_forked_repos

    if matching_codes_count_2 >= 6:
        return []

    user_repos_url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(user_repos_url, headers=headers)

    if response.status_code == 200:
        repos = response.json()
        code_repos = []

        for repo in repos:
            repo_name = repo['name']
            #print(f"Checking repository: {repo_name}")
            if repo['fork']:
                continue  # Skip forked repositories

            if any(contains_code_language(username, repo_name, lang) for lang in ['Python', 'Java']):
                #print(f"Repository {repo_name} contains relevant code.")
                code_repos.append(repo_name)
            else:
                pass
                #print(f"Repository {repo_name} does not contain relevant code.")

            if matching_codes_count_2 >= 6:
                break

        code_files = []
        for repo_name in code_repos:
            matching_codes_count = 0
            files_from_repo = get_code_from_folder(username, repo_name, "")
            
            if files_from_repo:
                code_files.extend(files_from_repo)
                matching_codes_count_2 += len(files_from_repo)

                if matching_codes_count_2 >= 6:
                    break
        
        return code_files
    else:
        #print(f"Error: Unable to fetch user repositories. Status Code: {response.status_code}")
        return None



def get_code_from_folder(username, repo_name, folder_path):
    global matching_codes_count

    if matching_codes_count >= 2:
        return []

    folder_contents_url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{folder_path}"
    response = requests.get(folder_contents_url, headers=headers)

    if response.status_code == 200:
        contents = response.json()
        code_files = []

        for item in contents:
            if 'type' in item and item['type'] == 'file' and is_valid_code_file(item['name']):
                file_content_url = item['download_url']
                file_content_response = requests.get(file_content_url, headers=headers)

                if file_content_response.status_code == 200:
                    content = file_content_response.text

                    if count_lines_of_code(content) > 50:
                        if has_core_logic(item['name'], content):
                            code_quality_metrics = get_code_metrics(content, item['name'])
                            code_files.append((content, code_quality_metrics))
                            #print(f"Found matching file: {item['name']}")
                            matching_codes_count += 1

                            if matching_codes_count >= 2:
                                break
                    else:
                        pass
                        #print(f"Skipping file '{item['name']}' - Less than 50 lines of code.")
                else:
                    #print(f"Error: Unable to fetch file content. Status Code: {file_content_response.status_code}")
                    return None
            elif 'type' in item and item['type'] == 'dir':
                files_from_folder = get_code_from_folder(username, repo_name, item['path'])
                if files_from_folder:
                    code_files.extend(files_from_folder)

        return code_files
    else:
        #print(f"Error: Unable to fetch folder contents. Status Code: {response.status_code}")
        return None
   
def analyze_python_code(python_code):
    # Create a temporary file for the Python code
    with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as temp_file:
        temp_file.write(python_code.encode('utf-8'))
        temp_file_path = temp_file.name

    # Run pylint analysis
    pylint_command = [
        'pylint',
        '--output-format=text',
        '--reports=n',
        temp_file_path
    ]
    pylint_result = subprocess.run(pylint_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pylint_output = pylint_result.stdout
    pylint_errors = pylint_result.stderr

    # Run radon analysis for cyclomatic complexity
    radon_command = [
        'radon',
        'cc',
        temp_file_path,
        '-s',
        '-j'
    ]
    radon_result = subprocess.run(radon_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    radon_output = radon_result.stdout
    radon_errors = radon_result.stderr

    return pylint_output, pylint_errors, radon_output, radon_errors

def parse_pylint_output(output, errors):
    error_count = 0
    warning_count = 0
    code_smells = 0

    # Parse pylint output for errors, warnings, and code smells
    for line in output.splitlines():
        if line.startswith('E'):  # Error
            error_count += 1
        elif line.startswith('W'):  # Warning
            warning_count += 1
        elif line.startswith('C') or line.startswith('R') or line.startswith('F'):  # Code Smells
            code_smells += 1

    # Count any stderr as errors
    error_count += len(errors.splitlines())

    return error_count, warning_count, code_smells

def parse_radon_output(output):
    cyclomatic_complexity = 0
    complexity_values = re.findall(r'"complexity":\s(\d+)', output)
    if complexity_values:
        cyclomatic_complexity = max(map(int, complexity_values))

    return cyclomatic_complexity

def analyze_java_code_with_pmd(java_code):
    # Create a temporary file for the Java code
    with tempfile.NamedTemporaryFile(delete=False, suffix='.java') as temp_file:
        temp_file.write(java_code.encode('utf-8'))
        temp_file_path = temp_file.name

    # Path to PMD executable
    pmd_path = r'Replace with PMD Path'
    
    # Multiple rulesets
    ruleset_paths = r'Replace Ruleset Path'

    # Run PMD analysis
    command = [
        pmd_path,
        'check',
        '-d', temp_file_path,
        '-R', ruleset_paths,
        '-f', 'text'
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Output results
    pmd_output = result.stdout
    pmd_errors = result.stderr

    return pmd_output, pmd_errors

def parse_pmd_output(output, errors):
    error_count = 0
    warning_count = 0
    cyclomatic_complexity = 0
    code_smells = 0

    # Count errors in stderr
    error_count = len([line for line in errors.splitlines() if line])

    # Parse stdout for warnings and cyclomatic complexity
    for line in output.splitlines():
        if "CyclomaticComplexity:" in line:
            match = re.search(r'complexity of (\d+)', line)
            if match:
                complexity = int(match.group(1))
                cyclomatic_complexity = max(cyclomatic_complexity, complexity)

        # Assuming warnings are labeled as such in output, adjust if necessary
        if "WARNING" in line:
            warning_count += 1

        # Any violation that isn't specifically categorized as an error or warning is a code smell
        if ":" in line and "CyclomaticComplexity" not in line:
            code_smells += 1

    return {
        "error_count": error_count,
        "warning_count": warning_count,
        "cyclomatic_complexity": cyclomatic_complexity,
        "code_smells": code_smells
    }

def get_code_metrics(content, filename):
    #print(f"Getting code metrics for file: {filename}")
    if filename.endswith('.py'):
        # Analyze Python code using pylint and radon
        pylint_output, pylint_errors, radon_output, radon_errors = analyze_python_code(content)
        
        # Parse pylint and radon outputs
        error_count, warning_count, code_smells = parse_pylint_output(pylint_output, pylint_errors)
        cyclomatic_complexity = parse_radon_output(radon_output)
        
        metrics = {
            "error_count": error_count,
            "warning_count": warning_count,
            "cyclomatic_complexity": cyclomatic_complexity,
            "code_smells": code_smells
        }
        #print(f"Calculated metrics for Python file: {metrics}")
        return metrics
    elif filename.endswith('.java'):
        # Analyze Java code using PMD
        pmd_output, pmd_errors = analyze_java_code_with_pmd(content)
        
        # Parse PMD output
        metrics = parse_pmd_output(pmd_output, pmd_errors)
        
        #print(f"Calculated metrics for Java file: {metrics}")
        return metrics
    else:
        #print("Unsupported file type.")
        return {
            "error_count": 0,
            "warning_count": 0,
            "cyclomatic_complexity": 0,
            "code_smells": 0
        }

def calculate_averages(metrics_list):
    total_files = len(metrics_list)
    if total_files == 0:
        return {
            "average_error_count": 0,
            "average_warning_count": 0,
            "average_cyclomatic_complexity": 0,
            "average_code_smells": 0
        }

    total_error_count = sum(m["error_count"] for m in metrics_list)
    total_warning_count = sum(m["warning_count"] for m in metrics_list)
    total_cyclomatic_complexity = sum(m["cyclomatic_complexity"] for m in metrics_list)
    total_code_smells = sum(m["code_smells"] for m in metrics_list)

    return {
        "average_error_count": total_error_count / total_files,
        "average_warning_count": total_warning_count / total_files,
        "average_cyclomatic_complexity": total_cyclomatic_complexity / total_files,
        "average_code_smells": total_code_smells / total_files
    }

def calculate_activity(commit_dates):
    now = datetime.now()
    commit_dates = [datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ') for date in commit_dates if date]
    days_active = len(set(date.date() for date in commit_dates))
    commits_per_day = len(commit_dates) / days_active if days_active else 0
    return commits_per_day, days_active

def check_documentation(username, repo_name):
    contents_url = f"{base_url}/repos/{username}/{repo_name}/contents/"
    contents_response = requests.get(contents_url, headers=headers)
    
    if contents_response.status_code != 200:
        #print(f"Error fetching contents for repository {repo_name}. Status Code: {contents_response.status_code}")
        return None
    
    contents = contents_response.json()
    
    for content in contents:
        if content['name'].lower() == "readme.md":
            return True
    
    return False

# Helper function to fetch repository languages concurrently
def fetch_languages(username, repo_name):
    languages_url = f"{base_url}/repos/{username}/{repo_name}/languages"
    lang_response = requests.get(languages_url, headers=headers)
    
    if lang_response.status_code == 200:
        return lang_response.json()
    return {}

# Parallelized version of list_languages
def list_languages(username):
    repos_url = f"{base_url}/users/{username}/repos"
    repos_response = requests.get(repos_url, headers=headers)
    
    if repos_response.status_code != 200:
        return None
    
    repos = repos_response.json()
    languages = {}
    
    # Parallelize fetching of languages for each repo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_repo = {executor.submit(fetch_languages, username, repo['name']): repo for repo in repos}
        
        for future in concurrent.futures.as_completed(future_to_repo):
            repo_languages = future.result()
            for lang, count in repo_languages.items():
                languages[lang] = languages.get(lang, 0) + count
    
    return languages

# Helper function to fetch commits for a single repo
def fetch_commits(username, repo_name):
    commits_url = f"{base_url}/repos/{username}/{repo_name}/commits"
    commits_response = requests.get(commits_url, headers=headers)
    
    if commits_response.status_code == 200:
        return commits_response.json()
    return []

# Parallelized version of check_activity
def check_activity(username):
    repos_url = f"{base_url}/users/{username}/repos"
    repos_response = requests.get(repos_url, headers=headers)
    
    if repos_response.status_code != 200:
        return None
    
    repos = repos_response.json()
    commit_dates = []
    
    # Parallelize fetching of commits for each repo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_repo = {executor.submit(fetch_commits, username, repo['name']): repo for repo in repos}
        
        for future in concurrent.futures.as_completed(future_to_repo):
            commits = future.result()
            for commit in commits:
                commit_date = commit['commit']['author']['date']
                commit_dates.append(commit_date)
    
    return calculate_activity(commit_dates)

# Parallelized version of list_contributions
def list_contributions(username):
    repos_url = f"{base_url}/users/{username}/repos"
    repos_response = requests.get(repos_url, headers=headers)
    
    if repos_response.status_code != 200:
        return None
    
    repos = repos_response.json()
    contributions = []
    
    # Parallelize fetching contributions for forked repos
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_repo = {executor.submit(lambda repo: repo if repo['fork'] else None, repo): repo for repo in repos}
        
        for future in concurrent.futures.as_completed(future_to_repo):
            repo = future.result()
            if repo and repo['owner']['login'] != username:
                contributions.append(repo['full_name'])
    
    return contributions

# Helper function to check the structure of a single repo
def fetch_repo_structure(username, repo_name):
    contents_url = f"{base_url}/repos/{username}/{repo_name}/contents/"
    contents_response = requests.get(contents_url, headers=headers)
    
    if contents_response.status_code != 200:
        return None
    
    contents = contents_response.json()
    structure = {"has_readme": False, "has_tests": False}
    
    for content in contents:
        if content['name'].lower() == "readme.md":
            structure["has_readme"] = True
        elif content['type'] == "dir" and content['name'].lower() == "tests":
            structure["has_tests"] = True
    
    return structure

# Parallelized version of check_project_structure
def check_project_structure(username, repo_name):
    # Directly fetching the structure without parallelization, but can be integrated if needed
    return fetch_repo_structure(username, repo_name)

def fetch_topics(username, repo_name):
    topics_url = f"{base_url}/repos/{username}/{repo_name}/topics"
    topics_response = requests.get(topics_url, headers={**headers, "Accept": "application/vnd.github.mercy-preview+json"})
    
    if topics_response.status_code != 200:
        return "Unknown"
    
    return topics_response.json().get('names', [])

def fetch_repo_contents(username, repo_name):
    contents_url = f"{base_url}/repos/{username}/{repo_name}/contents/"
    contents_response = requests.get(contents_url, headers=headers)
    
    if contents_response.status_code != 200:
        return []
    
    return contents_response.json()

def check_ml_files(content):
    # Download the file and check for ML keywords
    file_content_response = requests.get(content['download_url'])
    if file_content_response.status_code == 200:
        file_content = file_content_response.text
        if any(keyword in file_content for keyword in ["tensorflow", "keras", "scikit-learn", "pytorch", "training", "testing"]):
            return True
    return False

def identify_project_type(username, repo_name):
    # Define topic sets for different project types
    web_related_topics = {"web", "web-development", "django", "flask", "react", "vue", "angular", "html", "css", "javascript"}
    ml_related_topics = {"machine-learning", "deep-learning", "tensorflow", "pytorch", "keras", "scikit-learn"}
    devops_related_topics = {"devops", "docker", "kubernetes", "ci-cd", "terraform", "ansible"}
    
    # Initialize variables for flags
    has_web_topic = has_ml_topic = has_devops_topic = False
    has_web_files = has_ml_files = has_devops_files = False
    
    # Run parallel requests for topics and repo contents
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_topics = executor.submit(fetch_topics, username, repo_name)
        future_contents = executor.submit(fetch_repo_contents, username, repo_name)
        
        topics = future_topics.result()
        contents = future_contents.result()
        
        # Check if any topics match the predefined sets
        has_web_topic = any(topic in web_related_topics for topic in topics)
        has_ml_topic = any(topic in ml_related_topics for topic in topics)
        has_devops_topic = any(topic in devops_related_topics for topic in topics)
        
        # Check the files in the repo contents concurrently
        future_ml_files = [executor.submit(check_ml_files, content) for content in contents 
                           if content['name'].lower().endswith(('.py', '.ipynb')) or content['name'].lower() in ["train.py", "predict.py", "test.py", "model.py"]]
        
        for future in concurrent.futures.as_completed(future_ml_files):
            if future.result():
                has_ml_files = True
                break
        
        for content in contents:
            name_lower = content['name'].lower()
            
            # Check for common web-based project files
            if name_lower in ["package.json", "index.html", "app.py", "settings.py", "urls.py"]:
                has_web_files = True
            
            # Check for common DevOps-related files
            elif name_lower in ["dockerfile", "jenkinsfile", "terraform.tf", "kubernetes.yaml"]:
                has_devops_files = True

    # Determine project type based on topics and files
    if has_web_topic and has_ml_topic or has_web_files and has_ml_files:
        return "Web-Based with ML"
    elif has_web_topic or has_web_files:
        return "Web-Based"
    elif has_ml_topic or has_ml_files:
        return "Machine Learning"
    elif has_devops_topic or has_devops_files:
        return "DevOps"
    else:
        return "Unknown"
# Parallelized version of fetch_repo_details
def fetch_repo_details(username, repo_name):
    repo_url = f"{base_url}/repos/{username}/{repo_name}"
    repo_response = requests.get(repo_url, headers=headers)
    
    if repo_response.status_code != 200:
        return None
    
    repo_data = repo_response.json()
    forks_count = repo_data.get('forks_count', 0)
    watchers_count = repo_data.get('watchers_count', 0)
    stars_count = repo_data.get('stargazers_count', 0)
    
    # Parallel fetching of badges and repo details
    readme_url = f"{base_url}/repos/{username}/{repo_name}/readme"
    readme_response = requests.get(readme_url, headers=headers)
    
    badges = []
    if readme_response.status_code == 200:
        readme_data = readme_response.json()
        readme_content = requests.get(readme_data['download_url']).text
        badges = [line for line in readme_content.split('\n') if '![badge]' in line]
    
    return forks_count, watchers_count, stars_count, badges

# Parallelized version of fetch_user_pull_requests
def fetch_user_pull_requests(username):
    pull_requests_url = f"{base_url}/search/issues"
    params = {'q': f'is:pr author:{username}', 'per_page': 100}
    pull_requests_response = requests.get(pull_requests_url, headers=headers, params=params)
    
    if pull_requests_response.status_code != 200:
        return []
    
    return pull_requests_response.json().get('items', [])

def fetch_repo_details_parallel(username, repo):
    # Function to fetch details for a single repo
    repo_name = repo['name']
    forks_count, watchers_count, stars_count, badges = fetch_repo_details(username, repo_name)
    structure = check_project_structure(username, repo_name)
    project_type = identify_project_type(username, repo_name)
    documentation = check_documentation(username, repo_name)
    
    # Get repo languages (already parallelized in list_languages)
    repo_languages = fetch_languages(username, repo_name)
    
    return {
        "repo_name": repo_name,
        "structure": structure,
        "project_type": project_type,
        "documentation": documentation,
        "forks": forks_count,
        "watchers": watchers_count,
        "stars": stars_count,
        "badges": badges,
        "repo_languages": repo_languages
    }

def gather_non_code_quality_metrics(username, technologies_from_session, parsed_skills):
    global total_forked_repos
    # Fetch all repo languages concurrently
    languages = list_languages(username)
    total_usage = sum(languages.values())
    
    # Fetch commits and activity concurrently
    commits_per_day, days_active = check_activity(username)
    
    # Fetch contributions (open-source contributions)
    contributions = list_contributions(username)
    
    repos_url = f"{base_url}/users/{username}/repos"
    repos_response = requests.get(repos_url, headers=headers)
    
    if repos_response.status_code != 200:
        return None
    
    repos = repos_response.json()
    
    # Initialize soft skills tracking
    open_source_contributions = 0
    multiple_commits_projects = 0
    multi_language_projects = set()
    owned_projects_count = 0
    teamwork_contributions = 0
    total_forked_repos = 0
    
    analysis_results = []
    
    # Use ThreadPoolExecutor to parallelize fetching repo details
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each repo's details to be fetched in parallel
        future_to_repo = {executor.submit(fetch_repo_details_parallel, username, repo): repo for repo in repos}
        
        for future in concurrent.futures.as_completed(future_to_repo):
            repo_details = future.result()
            analysis_results.append(repo_details)
            
            # Collect soft skills-related data
            repo_name = repo_details['repo_name']
            repo = future_to_repo[future]
            
            # Count open source contributions for public repos where the user has contributed
            if not repo['private']:
                # Fetch repository contributors and check if the user is listed
                contributions_url = f"{base_url}/repos/{repo['owner']['login']}/{repo_name}/contributors"
                contributions_response = requests.get(contributions_url, headers=headers)
                
                if contributions_response.status_code == 200:
                    contributors = contributions_response.json()
                    if any(contributor['login'] == username for contributor in contributors):
                        open_source_contributions += 1

            # Count total forked repositories
            if repo['fork']:
                total_forked_repos += 1

            # Count multiple commits to one project
            commits_count = len(fetch_commits(username, repo_name))
            if commits_count > 1:
                multiple_commits_projects += 1

            # Track multiple languages across projects
            repo_languages = repo_details["repo_languages"]
            multi_language_projects.update(repo_languages.keys())

            # Count project ownership
            if repo['owner']['login'] == username:
                owned_projects_count += 1

            # Count teamwork contributions (Markdown files, issue comments)
            teamwork_contributions += len(fetch_user_pull_requests(username))
    
    # Aggregate user-level data
    pull_requests = fetch_user_pull_requests(username)
    total_repos = len(repos)
    documented_repos_count = sum(1 for result in analysis_results if result["documentation"])
    structured_repos_count = sum(1 for result in analysis_results if result["structure"] and result["structure"].get("has_tests", False))
    
    extra_technologies = set(languages.keys()) - set(technologies_from_session)
    extra_skills = set(languages.keys()) - set(parsed_skills)
    
    languages = {lang: (usage / total_usage * 100) for lang, usage in languages.items()}
    sorted_languages = dict(sorted(languages.items(), key=lambda item: item[1], reverse=True))
    
    return {
        "languages": sorted_languages,
        "activity": {"commits_per_day": commits_per_day, "days_active": days_active},
        "contributions": contributions,
        "analysis_results": analysis_results,
        "pull_requests": len(pull_requests),
        "total_repos": total_repos,
        "documented_repos": documented_repos_count,
        "undocumented_repos": total_repos - documented_repos_count,
        "structured_repos": structured_repos_count,
        "unstructured_repos": total_repos - structured_repos_count,
        "aggregate_metrics": {
            "forks": sum(result['forks'] for result in analysis_results),
            "watchers": sum(result['watchers'] for result in analysis_results),
            "stars": sum(result['stars'] for result in analysis_results),
            "badges": sum(len(result['badges']) for result in analysis_results)
        },
        "extra_technologies": list(extra_technologies),
        "extra_skills": list(extra_skills),
        "forks_from_other_repos": total_forked_repos,
        
        # Soft skills based on GitHub activities
        "soft_skills": {
            "familiarity_with_open_source": open_source_contributions > 0,
            "commitment_to_projects": multiple_commits_projects > 0,
            "fast_learner": len(multi_language_projects) > 1,
            "ownership_of_projects": owned_projects_count,
            "teamwork": teamwork_contributions > 0
        }
    }


def calculate_code_quality_metrics(username):
    code_files = get_code_from_repository(username)

    if code_files:
        metrics_list = []
        for content, metrics in code_files:
            metrics_list.append(metrics)

        # Calculate averages
        averages = calculate_averages(metrics_list)
        return averages
    else:
        return {}



def extract_github_username(github_link):
    # Define a regular expression pattern to match the username
    pattern = r'github\.com\/([a-zA-Z0-9-]+)'

    # Use re.search to find the match
    match = re.search(pattern, github_link)

    # Check if a match is found and return the username
    if match:
        return match.group(1)
    else:
        return None

def parse_pdf(file_path):
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

def extract_github_link_with_chatgpt(text):
    prompt = f"Extract the GitHub link from the following resume text: If no github link is present, return None\n\n{text}\n\nGitHub Link:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens = 50,
        temperature = 0.6,
        messages=[
            {"role": "system", "content": "You are advanced HR that extracts GitHub links."},
            {"role": "user", "content": prompt}
        ]
    )
    github_link = response.choices[0].message.content
    return github_link if github_link != 'None' else None

def extract_technologies_from_resume(text):
    prompt = f"Extract the technologies listed in the following resume text. Return a comma-separated list of technologies.\n\n{text}\n\nTechnologies:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.6,
        messages=[
            {"role": "system", "content": "You are a tech recruiter extracting technologies from resumes."},
            {"role": "user", "content": prompt}
        ]
    )
    technologies = response.choices[0].message.content.strip()
    return [tech.strip() for tech in technologies.split(',') if tech.strip()]



def calculate_non_code_quality_metrics(username,technologies_from_session,parsed_skills):
    # Perform the actual calculation of non-code quality metrics here
    non_code_quality_metrics = gather_non_code_quality_metrics(username, technologies_from_session,parsed_skills)
    #print(f"Metrics calculated for {username}: {non_code_quality_metrics}")  # Log metrics
    non_code_quality_data[username] = {'status': 'completed', 'metrics': non_code_quality_metrics}


def check_code_quality_status(request):
    username = request.GET.get('username', '')
    data = code_quality_data.get(username, {'status': 'pending', 'metrics': None})
    return JsonResponse(data)

def check_non_code_quality_status(request):
    username = request.GET.get('username', '')
    with lock:
        data = non_code_quality_data.get(username, {'status': 'pending'})
    return JsonResponse(data)

def no_github(request):
    return render(request, 'no_github.html')

def clear_previous_data(request):
    global code_quality_data, non_code_quality_data
    with lock:
        code_quality_data = {}
        non_code_quality_data = {}
    global matching_codes_count_2
    global matching_codes_count
    global total_forked_repos
    matching_codes_count_2 = 0
    matching_codes_count = 0
    total_forked_repos = 0
    request.session.pop('technologies', None)
    
def home(request):
    if request.method == 'POST':
        form = ResumeForm(request.POST, request.FILES)
        if form.is_valid():
            resume = form.save()
            file_name = 'demo.pdf'
            clear_previous_data(request)  # Clear previous data on new submission
            APIURL = "https://rest.rchilli.com/RChilliParser/Rchilli/parseResumeBinary"
            USERKEY = 'Rchilli Key'
            VERSION = '8.0.0'
            subUserId = 'Rchilli subuser ID'

            
            parsed_text = parse_pdf(resume.file.path)
            with open(resume.file.path, 'rb') as f:
                encoded_string = base64.b64encode(f.read()).decode('utf-8')
            github_link = extract_github_link_with_chatgpt(parsed_text)
            technologies = extract_technologies_from_resume(parsed_text)
            body = {
                    "filedata": encoded_string,
                    "filename": file_name,
                    "userkey": USERKEY,
                    "version": VERSION,
                    "subuserid": subUserId
                    }

            # Send the request to RChilli API
            headers = {'content-type': 'application/json'}
            response = requests.post(APIURL, data=json.dumps(body), headers=headers)

            # Handle the response
            if response.status_code == 200:
                resp = json.loads(response.text)
                resume_data = resp.get("ResumeParserData", {})

    
            # Extract Segregated Skills
                #skills = resume_data.get("SkillKeywords", {})
                segregated_skills = resume_data.get("SegregatedSkill", [])
    
    # Initialize lists to store soft skills and additional skills
                soft_skills = []
                additional_skills = []

    # Loop through the segregated skills and categorize them
                for skill in segregated_skills:
                    skill_type = skill.get("Type", "")
                    skill_name = skill.get("Skill", "")
        
                    if skill_type == "SoftSkill":
                        soft_skills.append(skill_name)
                    else:
                        additional_skills.append(skill_name)
            
            if github_link:
                username = extract_github_username(github_link)
                if username:
                    #skills = skills.split(",")  # Split by commas
                    #skills = [skill.strip() for skill in skills]
                    request.session['technologies'] = technologies
                    request.session['parsed_skills'] = additional_skills
                    request.session['soft_skills'] = soft_skills
                    return redirect('non_code_quality_info', username=username)
                else:
                    return redirect('no_github')
            else:
                return redirect('no_github')
    else:
        form = ResumeForm()
    return render(request, 'home.html', {'form': form})

def non_code_quality_info(request, username):
    technologies_from_session = request.session.get('technologies', [])
    parsed_skills = request.session.get('parsed_skills',[])
    soft_skills = request.session.get('soft_skills',[])

    threading.Thread(target=calculate_non_code_quality_metrics, args=(username, technologies_from_session,parsed_skills), daemon=True).start()
    return render(request, 'non_code_quality_info.html', {'username': username})

def results(request, username):
    if request.method == 'POST':
        with lock:
            code_quality_data.pop(username, None)
    
    non_code_quality_metrics = non_code_quality_data.get(username, {'status': 'pending'})
    
    if non_code_quality_metrics['status'] == 'pending':
        return render(request, 'non_code_quality_info.html', {'username': username})
    
    technologies = request.session.get('technologies', [])
    parsed_skills = request.session.get('parsed_skills',[])
    soft_skills = request.session.get('soft_skills',[])
    metrics = non_code_quality_metrics.get('metrics', {})
    
    if not metrics.get('analysis_results', []):
        return render(request, 'no_repositories.html')
    
    project_types = {
        "ml_based": sum(1 for result in metrics.get('analysis_results', []) if result['project_type'] == 'Machine Learning'),
        "web_based": sum(1 for result in metrics.get('analysis_results', []) if result['project_type'] == 'Web-Based'),
        "ci_cd_based": sum(1 for result in metrics.get('analysis_results', []) if result['project_type'] == 'CI/CD-Based'),
        "miscellaneous": sum(1 for result in metrics.get('analysis_results', []) if result['project_type'] == 'Unknown')
    }
    
    threading.Thread(target=run_code_quality_cal, args=(username,), daemon=True).start()
    
    data = code_quality_data.get(username, {'status': 'pending', 'metrics': None})

    return render(request, 'results.html', {
        'username': username,
        'metrics': non_code_quality_metrics['metrics'],
        'status': data['status'],
        'technologies': technologies,
        'parsed_skills': parsed_skills,
        'soft_skills': soft_skills,
        'project_types': project_types,
        'extra_technologies': metrics.get('extra_technologies', []),
        'extra_skills':metrics.get('extra_skills',[])
    })

def code_quality_results(request):
    username = request.GET.get('username', '')
    data = code_quality_data.get(username, {'status': 'pending', 'metrics': None})
    
    if not data['metrics']:
        return render(request, 'non_code_quality_info.html', {'username': username})

    return render(request, 'code_quality_results.html', {'username': username, 'metrics': data['metrics']})






