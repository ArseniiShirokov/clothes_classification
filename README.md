### Загрузка данных
1. Для доступа к данным необходимо примонтировать хранилище внутрь проекта:
```console
$ cd attributes
$ mkdir mnt
$ sudo mount -t cifs -o username=john.doe,domain=TVA //tva-storage-01.office.tevian.ru/tevian mnt/
```
2. Загрузите данные
```console
$ dvc pull
``` 
### Подготовка данных
Конфигурации наборов данных можно менять в tools/preproc_data/configs/ 
```console
$ python3 tools/preproc_data/preproc_data.py datasets=latest_train name="attributes_train"
$ python3 tools/preproc_data/preproc_data.py datasets=latest_val name="attributes_val"
$ python3 tools/preproc_data/preproc_data.py datasets=latest_spring_test name="metro_spring_test"
$ python3 tools/preproc_data/preproc_data.py datasets=latest_summer_test name="metro_summer_test"
``` 
### Проведение экспериментов
Конфигурации экспериментов можно менять в configs/ 
```console
$ cd docker/
$ ./run.sh
$ cd code
$ python3 train_and_test.py version=example
``` 

### Full body
Для full body все аналогично
```console
$ python3 tools/preproc_data/preproc_data.py datasets=fullbody_train name="fullbody_train" type="full_body"
$ python3 tools/preproc_data/preproc_data.py datasets=fullbody_val name="fullbody_val" type="full_body"
$ python3 tools/preproc_data/preproc_data.py datasets=fullbody_test name="fullbody_test" type="full_body"
``` 

Перед запуском нужно заменить configs/cinfig.yaml на configs/config_fullbody.yaml
```console
$ cd docker/
$ ./run.sh
$ cd code
$ python3 train_and_test.py version=example_fullbody
``` 

### Tools
В папке tools находятся вспомогательные скрипты:
* preproc_data -- переводит картинки в формат, использующийся моделями
* results_visualizator -- создает страничку с визуализацией сравнения результатов моделей
* data_mining -- таргетированный отбор сырых данных для разметки
* DatasetAnalyser -- создает страничуку с статистиками датасетов
