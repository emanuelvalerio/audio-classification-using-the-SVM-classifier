
% This code was developed by EMANUEL VALERIO PEREIRA

%% 
%  DESCRICAO DA BASE
% A base consta com 60 amostras onde 20 amostras sao da classe 1 que
% representa o audio da palavra um, 20 amostras sao da classe 2 que
% representa o audio da palavra dois, e mais 20 amostras da classe 3
% referente ao audio da palavra tres.
% A base consta com 17 atributos, a ultima coluna contabiliza 18 colunas e
% a ultima eh referente as classes 1 2 e 3. Os atributos extraidos dos
% sinais de audios, na qual compoem a base de dados foram:
%  average,stdMetric,skewnessMetric,kurtosisMetric,entropyMetric,
%  meanDeviation, Variance,cepstral,powerTotal,dominantFrequency, meanFrequency
% medianFrequency ,psdMean, psdMedian, psdMode, minMetric, rmsMetric.

%% 

clc;
clear all;
close all;

data = load('data.mat');
df = data.df;

%% 

% Abaixo, esta sendo realizada a normalizacao da base de dados extraida.
for i = 1:(size(df,2)-1)
    df(:,i) = (df(:,i)-mean(df(:,i)))/std(df(:,i)); % Normalizando os atributos
end

%% 


% Para evitar vies no modelo de treinamento, eh importante embaralhar a
% base de dados, abaixo isso esta sendo feito por meio da funcao randperm
% que gera valores aleatorios sem repeticao.
randomDb = randperm(size(df,1), size(df,1));
for i = 1:size(df,1)
    df(i,:) = df(randomDb(i),:);
end

%% 

% X = df(:,1:end-1); % features
X = df(:, 13:15); % only the estatistical features based in the power spectrum density are being used
y = df(:,end);     % classes

%% Valores para a função kernel
kernelFunctions = {'linear','polynomial','rbf'};

% Valores para os graus da funcao polinomial
degrees = [1, 2];

% Valores para a constante de relaxamento C
C_values = [1e-3, 1e-2, 1e-1, 1, 10, 100];

% Valores para o kernel scale
kernelScales = [1e-3, 1e-2, 1e-1, 1, 10, 100];
numFolds = 3; % ou qualquer outro valor desejado
predictMatrix = [];

yPred = [];
bestAccuracy = 0;
bestParams = struct();  %  Struct para armazenar os melhores hiperparametros
%% 

% O Loop abaixo ira testar todas as combinacoes de hiperparametros
% possiveis baseado na acuracia, no fim ira retornar a melhor acuracia
% obtida e os melhores hiperparametros que possibilitaram essa melhor
% acuracia.

for k = 1:length(kernelFunctions)  % Loop que pecorre as funcoes kernel
    if strcmpi(kernelFunctions{k}, 'polynomial') % A comparacao existe por conta de que sera testado 2 graus da funcao polinomial
        for j = 1:length(degrees) % Nesse caso eh necessario um laco adicional para pecorres as possibilidades de graus da funcao polinomial
            for c = 1:length(C_values)  % Loop que pecorre os possiveis valores para a constante de relaxamento.
                for s = 1:length(kernelScales) % Loop que pecorre os possiveis valores do kernel scale

                    % Definir os hiperparametros atuais
                    currentKernel = kernelFunctions{k};
                    currentC = C_values(c);
                    currentKernelScale = kernelScales(s);
        
                    % Inicializando a variavel para armazenar a acuracia media da validação cruzada
                    foldAvgAccuracy = 0;
        
                    % Realizar validacao cruzada k-fold
                    cv = cvpartition(length(y), 'KFold', numFolds);
        
                    for fold = 1:numFolds
                        train_idx = cv.training(fold);
                        test_idx = cv.test(fold);
                        %  Dividindo os dados em treino e teste.
                        XTrain = X(train_idx, :);
                        yTrain = y(train_idx);
                        XTest = X(test_idx, :);
                        yTest = y(test_idx);
                        
                        
                        classes = unique(yTrain); % Obtendo as classes únicas presentes nos dados
                        predictMatrix = [];       %  Matriz de classes preditas pelo classificador
                        numClasses = length(classes); % Numero de classes
                        
                        % Inicializando a matriz de contagem das previsoes
                        countPredictions = zeros(numClasses, numClasses);
                        
                        % Para a implementacao do metodo one vs one do
                        % classificador svm, deve-se treinar as classes
                        % binarias em todas as suas permutacoes no caso
                        % como eh 3 classes, as possibilidades sao 1 2, 1 3
                        % e 2 3, abaixo realiza-se os lacos para
                        % possibilitar essa dinamica.
                        for classA = 1:numClasses-1
                            for classB = classA+1:numClasses
                                try
                                    % Nesse caso, deve-se usar para treino
                                    % e teste apenas as classes binarias
                                    % atuais do laco, excluindo as demais,
                                    % abaixo uma nova base de treino eh
                                    % definida com base nas classes atuais
                                    % do laco.

                                    XbinaryTrain = XTrain(yTrain == classA | yTrain == classB,:);
                                    ybinaryTrain = yTrain(yTrain == classA | yTrain == classB,:);

                                    
                                    % Treinamento do modelo SVM passando os
                                    % atuais hiperparametros 
                                    % A funcao fitcsvm realiza o treinamento do
                                    % modelo considerando classes binarias.
                                    svmModel = fitcsvm(XbinaryTrain,ybinaryTrain, 'KernelFunction',currentKernel, 'PolynomialOrder', degrees(j), 'BoxConstraint', currentC, 'KernelScale', currentKernelScale);
            
                                    predictions = predict(svmModel, XTest); % Testando o modelo na base de teste
                                catch
                                    % Ignorar o erro caso ocorra
                                    continue;
                                end
                                  
                                predictMatrix = [predictMatrix, predictions]; % armazena em uma matriz os vetores de predicao de cada combinacao do modelo one x one
                            end
                        end       
                        
                        predictions = predictMatrix; 
                        numberEachClass = zeros(size(predictions,1),length(classes));
                        
                        % Abaixo propoe-se um modelo de analise da
                        % quantidade de predicoes cada classe recebeu, esse
                        % modelo eh generico, para qualquer quantidade de
                        % classes ele eh capaz de identificar se houve
                        % empate a partir da quantidade de pontos que cada
                        % classe recebeu.

                        for i = 1:size(predictions,1)
                            for jj = 1:length(classes)
                                if isempty(numel(intersect([classes(jj)], predictions(i,:)))) % verifica a qauntidade de cada classe que foi predita para cada amostra.
                                    numberEachClass(i,jj) = 0; % Se nao ha nenhuma classe para as classes predita, adiciona-se 0.
                                else
                                    numberEachClass(i,jj) = sum(predictions(i,:) == classes(jj)); % caso o vetor nao seja vazio, significa que classes que foram preditas
                                                                                                  % Nesse caso, realiza-se a contagem de cada vez que uma classe foi predita.
                                end
                            end
                        end
                        
                        predictMatrix = [];

                        % Abaixo, baseado na quantidade de pontos que cada
                        % classe obteve, sera feito a predicao multiclasse,
                        % ou seja, a classe predita sera aquela que obteve
                        % mais pontos na predicao binaria, abaixo o
                        % problema de empate eh tratado escolhendo a classe
                        % predita aleatoriamente.

                        for i = 1:size(numberEachClass,1)
                            sampleMax = max(numberEachClass(i,:));
                            
                            % Encontrar os indices dos valores maiores, ou
                            % seja, retornara um vetor de maximos, esses
                            % maximos sao as classes que foram mais
                            % preditas no modelo. 
                            indexMax = find(numberEachClass(i,:) == sampleMax);
                            vectorMax = numberEachClass(i,indexMax);
                            
                            if length(vectorMax) == 1
                                ypred(i) = indexMax; % se esse vetor de maximos eh unitario, significa que nao houve empate, logo a classe predita sera a classe que obteve mais pontos.
                                % indexMax eh a coluna que teve mais
                                % pontos, ou seja, cada coluna eh a classe,
                                % logo eh a classe que obteve mais pontos.
                            else
                                % Caso existir empate entre classes, a
                                % classe predita sera feita aleatoriamente
                                % tal como esta ocorrendo abaixo.
                                idx = randi(numel(vectorMax));
                                choosedClass = indexMax(idx);
                                ypred(i) = choosedClass; % classe predita escolhida de forma aleatoria no caso de empate.
                            end
                        end
                        
                        % Calculando a acuracia para o fold atual e adiciona-o a media
                        accuracyCurrent(fold) = sum(ypred' == yTest) / length(yTest);
                    end
        
                    % Calculando a acuracia media entre todas as acuracias
                    % de cada folds.
                    foldAvgAccuracy = mean(accuracyCurrent); % acuracia media dos k-folds
        
                    % Verificando se essa combinacao de hiperparametros eh
                    % a melhor basenado-se na acuracia.
                    if foldAvgAccuracy > bestAccuracy
                        bestAccuracy = foldAvgAccuracy;
                        bestParams.kernel = currentKernel;
                        best_degree = degrees(j);
                        bestParams.C = currentC;
                        bestParams.kernel_scale = currentKernelScale;
                    end
                end
            end
        end

    else % Caso a funcao do kernel nao seja polinomial, nao ha necessidade de um laco a mais para calcular as possibilidades dos graus.
        % Logo o codigo segue a mesma linha, apenas sem a presenca do laco
        % referente aos graus do polinomio

        for c = 1:length(C_values) % Loop que pecorre os possiveis valores para a constante de relaxamento.
            for s = 1:length(kernelScales) % Loop que pecorre os possiveis valores do kernel scale
               % Definir os hiperparametros atuais
                currentKernel = kernelFunctions{k};
                currentC = C_values(c);
                currentKernelScale = kernelScales(s);
        
                % Inicializando a variavel para armazenar a acuracia media da validação cruzada
                foldAvgAccuracy = 0;
        
                % Realizar validacao cruzada k-fold
                cv = cvpartition(length(y), 'KFold', numFolds);
        
                for fold = 1:numFolds
                    train_idx = cv.training(fold);
                    test_idx = cv.test(fold);
                    %  Dividindo os dados em treino e teste.
                    XTrain = X(train_idx, :);
                    yTrain = y(train_idx);
                    XTest = X(test_idx, :);
                    yTest = y(test_idx);
                    
                    
                    classes = unique(yTrain); % Obtendo as classes únicas presentes nos dados
                    predictMatrix = [];       % Matriz de classes preditas pelo classificador
                    numClasses = length(classes);
                    
                    % Inicializando a matriz de contagem das previsoes
                    countPredictions = zeros(numClasses, numClasses);
                    
                    % Para a implementacao do metodo one vs one do
                    % classificador svm, deve-se treinar as classes
                    % binarias em todas as suas permutacoes no caso
                    % como eh 3 classes, as possibilidades sao 1 2, 1 3
                    % e 2 3, abaixo realiza-se os lacos para
                    % possibilitar essa dinamica.
        
                    for classA = 1:numClasses-1
                        for classB = classA+1:numClasses
                            try
                                % Nesse caso, deve-se usar para treino
                                % e teste apenas as classes binarias
                                % atuais do laco, excluindo as demais,
                                % abaixo uma nova base de treino eh
                                % definida com base nas classes atuais
                                % do laco.
                                XbinaryTrain = XTrain(yTrain == classA | yTrain == classB,:);
                                ybinaryTrain = yTrain(yTrain == classA | yTrain == classB,:);

                                % Treinamento do modelo SVM passando os
                                % atuais hiperparametros 
                                % A funcao fitcsvm realiza o treinamento do
                                % modelo considerando classes binarias.
                                svmModel = fitcsvm(XbinaryTrain, ybinaryTrain, 'KernelFunction',currentKernel, 'BoxConstraint', currentC, 'KernelScale', currentKernelScale);
        
                                predictions = predict(svmModel, XTest);  % realizando as predicoes com base no modelo do classificador svm
                                
                            catch
                                % Ignorar o erro caso ocorra
                                continue;
                            end
       
                            predictMatrix = [predictMatrix, predictions];  % armazena em uma matriz os vetores de predicao de cada combinacao do modelo one x one

                            
                        end
                    end       
                    
                    predictions = predictMatrix;
                    numberEachClass = zeros(size(predictions,1),length(classes));
                    
                    % Abaixo propoe-se um modelo de analise da
                    % quantidade de predicoes cada classe recebeu, esse
                    % modelo eh generico, para qualquer quantidade de
                    % classes ele eh capaz de identificar se houve
                    % empate a partir da quantidade de pontos que cada
                    % classe recebeu.
                    for i = 1:size(predictions,1)
                        for j = 1:length(classes)
                            if isempty(numel(intersect([classes(j)], predictions(i,:)))) % verifica a qauntidade de cada classe que foi predita para cada amostra.
                                numberEachClass(i,j) = 0; % Se nao ha nenhuma classe para as classes predita, adiciona-se 0.
                            else
                                numberEachClass(i,j) = sum(predictions(i,:) == classes(j)); % caso o vetor nao seja vazio, significa que classes que foram preditas
                                                                                            % Nesse caso, realiza-se a contagem de cada vez que uma classe foi predita.
                            end
                        end
                    end
                    
                    predictMatrix = [];
                    for i = 1:size(numberEachClass,1)
                        sampleMax = max(numberEachClass(i,:));
                        % Encontrar os indices dos valores maiores, ou
                        % seja, retornara um vetor de maximos, esses
                        % maximos sao as classes que foram mais
                        % preditas no modelo.

                        indexMax = find(numberEachClass(i,:) == sampleMax);
                        vectorMax = numberEachClass(i,indexMax);
                        
                        if length(vectorMax) == 1
                            ypred(i) = indexMax; % se esse vetor de maximos eh unitario, significa que nao houve empate, logo a classe predita sera a classe que obteve mais pontos.
                            % indexMax eh a coluna que teve mais
                            % pontos, ou seja, cada coluna eh a classe,
                            % logo eh a classe que obteve mais pontos.
                        else
                            % Caso existir empate entre classes, a
                            % classe predita sera feita aleatoriamente
                            % tal como esta ocorrendo abaixo.
                            idx = randi(numel(vectorMax));
                            choosedClass = indexMax(idx);
                            ypred(i) = choosedClass; % classe predita escolhida de forma aleatoria no caso de empate.
                        end
                    end
                    
                    % Calculando a acuracia para o fold atual e adiciona-o a media
                    accuracyCurrent(fold) = sum(ypred' == yTest) / length(yTest);
                end
        
                % Calcular a acuracia media atraves de todos os folds
                foldAvgAccuracy = mean(accuracyCurrent);
        
                % Verificando se essa combinacao de hiperparametros eh
                % a melhor basenado-se na acuracia.
                if foldAvgAccuracy > bestAccuracy
                    bestAccuracy = foldAvgAccuracy;
                    bestParams.kernel = currentKernel;
                    best_degree = 0; % Para kernels diferentes de polinomial, definimos o grau como 0
                    bestParams.C = currentC;
                    bestParams.kernel_scale = currentKernelScale;
                end
            end
        end
    end
end

%% Exibindo os melhores hiperparametros encontrados para a melhor accuracia calculada.
if strcmpi(bestParams.kernel, 'polynomial')
    fprintf('Melhores hiperparâmetros:\n');
    fprintf('best accuracy: %f\n', bestAccuracy);
    fprintf('Função kernel: %s\n', bestParams.kernel);
    disp(['Grau: ', num2str(best_degree)]);
    fprintf('Constante de relaxamento C: %.3f\n', bestParams.C);
    fprintf('Kernel scale: %.3f\n', bestParams.kernel_scale);
else
    fprintf('Melhores hiperparâmetros:\n');
    fprintf('best accuracy: %f\n', bestAccuracy);
    fprintf('Função kernel: %s\n', bestParams.kernel);
    fprintf('Constante de relaxamento C: %.3f\n', bestParams.C);
    fprintf('Kernel scale: %.3f\n', bestParams.kernel_scale);
end

