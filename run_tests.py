import os
os.environ['API_PASSWORD'] = 'testpassword123'
import subprocess
subprocess.run(['python', '-m', 'pytest', 'tests', '-q', '--tb=short'])