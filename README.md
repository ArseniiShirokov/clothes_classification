### Загрузка данных
1. Загрузите данные:

   [Download datasets](https://drive.google.com/drive/folders/1-Azvod5qbqpdo0CKijjZH-eQf-JYWaDE?usp=sharing)
### Подготовка данных
Конфигурации наборов данных можно менять в tools/preproc_data/configs/ 
```console
$ python3 tools/preproc_data/preproc_data.py datasets=fashion-mnist-train name="train"
$ python3 tools/preproc_data/preproc_data.py datasets=fashion-mnist-test name="test"
``` 
### Проведение экспериментов
Конфигурации экспериментов можно менять в configs/ 
```console
$ cd docker/
$ ./run.sh
$ cd code
$ python3 train_and_test.py version=example
```

### Tools
В папке tools находятся вспомогательные скрипты:
* preproc_data -- переводит картинки в формат, использующийся моделями
* results_visualizator -- создает страничку с визуализацией сравнения результатов моделей
* data_mining -- таргетированный отбор сырых данных для разметки
* DatasetAnalyser -- создает страничуку с статистиками датасетов
