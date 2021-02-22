# Planet Remote Sensing Tools

## Pull Docker Image
```bash
docker pull planetlabs/notebooks
docker tag planetlabs/notebooks planet-notebooks
```

## Create .env file and add Planet API Key
Create a .env file with the required variables. An [example file](.env.example) is provided, rename it to .env and add the appropriate values:

```bash
cd planet-remote-sensing-tools
cp .env.example .env
vi .env
```

## Run Docker Image
```bash
docker run -it --rm -p 8888:8888 -v $PWD:/home/jovyan/work --env-file=.env -e PL_API_KEY planet-notebooks
```

## Credits
* https://github.com/planetlabs/notebooks/