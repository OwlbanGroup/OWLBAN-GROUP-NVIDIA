import os
os.environ['PG_USER'] = ''
os.environ['PG_PASSWORD'] = ''
os.environ['API_PASSWORD'] = 'testpassword123'

if __name__ == "__main__":
    import main
    main.main()