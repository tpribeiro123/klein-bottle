    %
    % Implementacao da Garrafa de Klein para todas as Bases, Tamanhos e
    % Frequencias
    %
    % 2018-05-14
    %

    clc;
    clear;
    close all;

    warning('off','all')

    % Limpar variaveis: % (1) Sim (2) Nao
    limpaVar = 1;

    % Qtde de digitos
    digits(100);

    % Salvar os dados em arquivo .mat: (1) Sim ou (2) Nao
    salvar = 1;

    % Frequencias de Corte
    % Utilizado no laco para o corte de frequencias
    FreqCorteIni = 6;
    FreqCorteFim = 6;

    % Matriz de Tamanhos de Patchs a Projecao da Garrafa
    % 1-Artigo, 2-Fibonacci, 3-Sequencia, 4-Seq Pares, 5-Seq Impares, 6-
    matrizPatch = 1;

    switch matrizPatch
        case 1
            % patchs do artigo
            matPatchsize = [3 7 11 15 19];
        case 2
            % patchs fibonacci
            matPatchsize = [3 5 8 13 21 34 55];
        case 3
            % combina 13 patchs
            matPatchsize = 3:2:27;
        case 4
            % sequencia
            matPatchsize = 3:12;
        case 5
            % sequencia
            matPatchsize = [3 4 5];
    end

    distMaxima = distanciaMaxima(matPatchsize);

    % Path para execucao
    zz = strsplit(pwd,'/');
    idx = find(ismember(lower(zz),'owncloud'));
    if idx ~= 0
        iniciaPath = '';
        for i=1:idx
            iniciaPath = strcat(iniciaPath,zz{i},'/');
        end
    else
        fprintf('ERRO NAO EXISTE BASE DE IMAGENS!!!');
        return;
    end
    clear zz idx i;

    ticTudoF = tic;
    cl = fix(clock);

    for iM = 11:11
        % Bases de Imagens
        % (1)Brodatz (2)Vistex (3)Outex (4)KTH_TIPS (5)CUReT (6)UIUCTex
        % (7)Pollen (8)SIPI (9)Sport_Event (10)Swedish_Leaf
        % Artigo: 4, 5 e 6

        % Base com subdiretorios
        % (0) nao   (1) sim
        bSubdir = 0;

        % Se existem qtdes diferentes de imagens por classe
        % (0) Nao (1) Sim
        TipoTreino = 0;

        switch iM
            case 1
                baseN = 'Brodatz';
                base = [baseN '/'];
                classes = 111;
                imagens = 16;
            case 2
                baseN = 'Vistex';
                base = [baseN '/'];
                classes = 54;
                imagens = 16;
            case 3
                baseN = 'Outex';
                base = [baseN '/'];
                classes = 68;
                imagens = 20;
            case 4
                baseN = 'KTH_TIPS';
                base = [baseN '/'];
                classes = 10;
                imagens = 81;
            case 5
                baseN = 'CUReT';
                base = 'CUReT_Mod/';
                classes = 61;
                imagens = 92;
            case 6
                baseN = 'UIUCTex';
                base = 'UIUCTex/';
                classes = 25;
                imagens = 40;
            case 7
                baseN = 'Pollen';
                base = 'pollen_grain_data_renamed/';
                classes = 30;
                imagens = 40; %variavel
                bSubdir = 1;
                TipoTreino = 1;
            case 8
                baseN = 'SIPI';
                base = [baseN '/'];
                classes = 13;
                imagens = 7;
            case 9
                baseN = 'Sport_Event';
                base = 'event_img/';
                classes = 8;
                imagens = 137; %variavel
                bSubdir = 1;
                TipoTreino = 1;
            case 10
                baseN = 'Swedish_Leaf';
                base = [baseN '/'];
                classes = 15;
                imagens = 75;
                bSubdir = 1;
            case 11
                baseN = 'alot_grey2';
                base = [baseN '/'];
                classes = 250;
                imagens = 100;
                bSubdir = 1;
        end

        path_base_name = [iniciaPath 'Bases_Imagens/' base];

        dCell = busca_nomes_imagens2(path_base_name,bSubdir);
        tamanhod = length(dCell);

        % define Matriz de Rotulos das Classes
        MatLabelS=[;];
        if (TipoTreino == 0)
            for k=1:classes
                MatLabelS = [MatLabelS k*ones(1,imagens)];
            end
            MatLabelS = MatLabelS;
        else
            for k=1:size(dCell,1)
                MatLabelS = [MatLabelS; dCell(k).class];
            end
        end

        clear k;
        
        if isempty(gcp('nocreate'))
            parpool('local');
        end

    %
    % Projetando na Garrafa de Klein
    %
        ticTudo = tic;

        fprintf('\nIniciando Projecao...');

    %    for imgBase=1:tamanhod
        parfor imgBase=1:tamanhod
            ticTotal = tic;
            fprintf('\nBase: %s - Imagem %d...',baseN,imgBase);

            for patchsizeNumero=1:size(matPatchsize,2)

                ticPatch = tic;

                DescartouTodos = 'N';

                % define o tamanho do path
                patchsize = matPatchsize(patchsizeNumero);

                nome = dCell(imgBase,1).name;
                if isfield(dCell,'dir')
                    nomeAbre=strcat(path_base_name,dCell(imgBase,1).dir,'/',nome);
                else
                    nomeAbre = [path_base_name nome];
                end
                input=imread(nomeAbre);

                fprintf('\nPatch(%dx%d) - Imagem(%d): %s ',patchsize,patchsize,imgBase,nome);
                if limpaVar == 1
                    nome=[];
                end
                
                [h, w, Cor] = size(input);

                if (Cor == 3)
                    input = rgb2gray(input);
                end

                % We add 1 to each entry in the matrix, and following Weber’s law
                % take entry-wise natural logarithms.
                input = input + 1;

                % Recorta Patchs
                totalPatch = fix(w/patchsize) * fix(h/patchsize);

                % Selecionando 5000 patchs (aleatoriamente)
                sel_patch = [;];
                if totalPatch <= 5000
                sel_patch = (1:totalPatch);
                else
                    sel_patch = randperm(totalPatch,5000);
                end

                % Posicao do Patch na Imagem
                Pos_Patch_Imagem=[;];
                for i = 1:size(sel_patch,2)
                    Pos_Patch_Imagem(:,i) = retPosicao(sel_patch(i),patchsize,w);
                end

                % Salva os Patchs na matriz
                todos_patch = [;];
                for k = 1: size(Pos_Patch_Imagem,2)
                    tmp_img = input(Pos_Patch_Imagem(1,k):Pos_Patch_Imagem(1,k)+patchsize-1,Pos_Patch_Imagem(2,k):Pos_Patch_Imagem(2,k)+patchsize-1);
                    todos_patch(k,:) = tmp_img(:);
                end

                if limpaVar == 1
                    tmp_img=[];
                    input=[];
                    Ttodos_patch=[];
                end

                % Aplicando o log nos patchs
                res_patch = log(todos_patch);

                if limpaVar == 1
                    todos_patch=[];
                    k=[];
                    i=[];
                end

                % DNORMA
                %Construcao da D-norma
                DnormaPatchs=[;];
                dnorma1=zeros(patchsize-1,patchsize-1,size(res_patch,1));
                dnorma2=zeros(patchsize,patchsize,size(res_patch,1));
                dnorma3=zeros(patchsize,patchsize,size(res_patch,1));
                for ks=1:size(res_patch,1)
                    temp_res_patch = reshape(res_patch(ks,:),patchsize,patchsize);
                    for is=1:patchsize-1
                        for js=1:patchsize-1
                        dnorma1(is,js,ks)=(temp_res_patch(is,js)-temp_res_patch(is+1,js))^2+(temp_res_patch(is,js)-temp_res_patch(is,js+1))^2;
                        dnorma2(is,patchsize,ks)=(temp_res_patch(is,patchsize)-temp_res_patch(is+1,patchsize))^2;
                        dnorma3(patchsize,js,ks)=(temp_res_patch(patchsize,js)-temp_res_patch(patchsize,js+1))^2;
                        end
                    end
                end
                for ks=1:size(res_patch,1)
                DnormaPatchs(ks,:)=[(sqrt(sum(sum(dnorma1(:,:,ks)))+sum(sum(dnorma2(:,:,ks)))+sum(sum(dnorma3(:,:,ks))))) ks sel_patch(ks)];
                end

                if limpaVar == 1
                    dnorma1=[]; dnorma2=[]; dnorma3=[]; temp_res_patch=[];
                    ks=[];is=[]; js=[];
                end

                % Selecionando os 1000 maiores patchs
                for i=size(DnormaPatchs,1):-1:1
                    if DnormaPatchs(i,1) <= 0.01
                        DnormaPatchs(i,:) = [];
                    end
                end

                porMaiores = 1000;
                % As 1000 maiores Dnormas
                DnormaPatchs_Maiores = sortrows(DnormaPatchs,1);
                comecaCorte = size(DnormaPatchs_Maiores,1)-(porMaiores-1);  %999;
                if size(DnormaPatchs_Maiores,1) < porMaiores  %1000
                    comecaCorte = 1;
                end
                DnormaPatchs_Maiores = DnormaPatchs_Maiores(comecaCorte:end,:);

                if limpaVar == 1
                    comecaCorte=[]; DnormaPatchs=[]; i=[];
                end

                novo_res_patch=[;];
                for i=1:size(DnormaPatchs_Maiores,1)
                    novo_res_patch(i,:) = res_patch(DnormaPatchs_Maiores(i,2),:);
                end

                if limpaVar == 1
                    DnormaPatchs_Maiores=[]; Pos_Patch_Imagem=[];
                    res_patch=[]; sel_patch=[]; i=[];
                end

                %Tirando a media, centralizando e normalizando
                dnorma_patchy=[;];
                dnorma1=zeros(patchsize-1,patchsize-1,size(res_patch,1));
                dnorma2=zeros(patchsize,patchsize,size(res_patch,1));
                dnorma3=zeros(patchsize,patchsize,size(res_patch,1));
                temp_res_patch_Todos=[;];
                for ks=1:size(novo_res_patch,1)
                    temp_res_patch = reshape(novo_res_patch(ks,:),patchsize,patchsize)-1/(patchsize)^2*mean(novo_res_patch(ks,:));
                    temp_res_patch_Todos(:,:,ks) = temp_res_patch(:)';
                    for is=1:patchsize-1
                        for js=1:patchsize-1
                        dnorma1(is,js,ks)=(temp_res_patch(is,js)-temp_res_patch(is+1,js))^2+(temp_res_patch(is,js)-temp_res_patch(is,js+1))^2;
                        dnorma2(is,patchsize,ks)=(temp_res_patch(is,patchsize)-temp_res_patch(is+1,patchsize))^2;
                        dnorma3(patchsize,js,ks)=(temp_res_patch(patchsize,js)-temp_res_patch(patchsize,js+1))^2;
                        end
                    end
                end
                for ks=1:size(novo_res_patch,1)
                    dnorma_patchy(ks,:)=sqrt(sum(sum(dnorma1(:,:,ks)))+sum(sum(dnorma2(:,:,ks)))+sum(sum(dnorma3(:,:,ks))));
                end

                res_patchnormalizado=[;];
                for ks2=1:size(novo_res_patch,1)
                    res_patchnormalizado(ks2,:)=(1/dnorma_patchy(ks2))*temp_res_patch_Todos(:,:,ks2);
                end

                if limpaVar == 1
                    dnorma1=[]; dnorma2=[]; dnorma3=[]; temp_res_patch=[];
                    dnorma_patchy=[]; ks=[]; is=[]; js=[]; ks2=[];
                    temp_res_patch_Todos=[];
                end

                % Patchs Finais Centralizados e Normalizados.
                final_patches = res_patchnormalizado;

                if limpaVar == 1
                    res_patchnormalizado=[];
                end

                % DeltaP(i,j)
                deltaPij = cell(patchsize,patchsize,size(final_patches,1));
                % HessP(i,j)
                hessPij = cell(patchsize,patchsize,size(final_patches,1));
                for ii=1:size(final_patches,1)
                    tmpPatch = reshape(final_patches(ii,:),patchsize,patchsize);
                    tmp_deltaPij = cell(patchsize,patchsize);
                    tmp_hessPij = cell(patchsize,patchsize);
                    for i=1:patchsize
                        for j=1:patchsize
                            if ((i==1) || (j==1) || (i==patchsize) || (j==patchsize))
                                tmp_deltaPij(i,j) = {[0 0]};
                                tmp_hessPij(i,j) = {[0 0; 0 0]};
                            else
                                % DeltaP
                                tmp_deltaPij(i,j) = {[((tmpPatch(i,j+1) - tmpPatch(i,j-1))/2) ((tmpPatch(i-1,j) - tmpPatch(i+1,j))/2)]};
                                % HessP
                                tmp_hessPij(i,j) = {[((tmpPatch(i,j+1) - (2*tmpPatch(i,j)) + tmpPatch(i,j-1))) ...
                                                    ((tmpPatch(i-1,j+1) - tmpPatch(i-1,j-1) + tmpPatch(i+1,j-1) - tmpPatch(i+1,j+1)) /4); ...
                                                    ((tmpPatch(i-1,j+1) - tmpPatch(i-1,j-1) + tmpPatch(i+1,j-1) - tmpPatch(i+1,j+1)) /4) ...
                                                    ((tmpPatch(i+1,j) - (2*tmpPatch(i,j)) + tmpPatch(i-1,j)))]};
                            end
                        end
                    end
                    deltaPij(:,:,ii) = tmp_deltaPij;
                    hessPij(:,:,ii) = tmp_hessPij;
                end

                if limpaVar == 1
                    tmp_deltaPij=[]; tmp_hessPij=[]; tmpPatch=[];
                    conta=[];
                    i=[]; j=[]; ii=[];
                end

                % >>>>>>>>>>>>>>>
                %>>>>>>>>>>>> r e t devem estar na borda para o calculo senao o valor e o deltaPzero
                % >>>>>>>>>>>>>>>
                % DeltaP(r,t)
                % deltaPij{1,1,1}(1) Px - derivada parcial de x
                % deltaPij{1,1,1}(2) Py - derivada parcial de y
                derivaParcialX = NaN(patchsize,patchsize,size(final_patches,1));
                derivaParcialY = NaN(patchsize,patchsize,size(final_patches,1));
                for k=1:size(final_patches,1)
                    for r=1:patchsize
                        for t=1:patchsize
                            if ((r==1 || r==patchsize) || (t==1 || t==patchsize))
                                if (t==1)
                                    j=2;
                                elseif t==patchsize
                                    j=patchsize-1;
                                end
                                if (r==1)
                                    i=2;
                                elseif (r==patchsize)
                                    i=patchsize-1;
                                end
                                tmpDeltaPrt = (cell2mat(deltaPij(i,j,k))') + (cell2mat(hessPij(i,j,k)) * ([t-j i-r]'));
                            else
                                tmpDeltaPrt = cell2mat(deltaPij(i,j,k))';
                            end
                            derivaParcialX(r,t,k) = tmpDeltaPrt(1);
                            derivaParcialY(r,t,k) = tmpDeltaPrt(2);
                        end
                    end
                end

                if limpaVar == 1
                    tmpDeltaPrt=[]; tmpHessPij=[];
                    k=[]; r=[]; t=[]; i=[]; j=[];
                    deltaPij=[]; hessPij=[];
                end

                % Calculo do Ap
                % 1 3
                % 2 4
                aP = NaN(size(final_patches,1),4);
                for i=1:size(final_patches,1)
                    aP1 = sum(sum(derivaParcialX(:,:,i).*derivaParcialX(:,:,i)));
                    aP2 = sum(sum(derivaParcialX(:,:,i).*derivaParcialY(:,:,i)));
                    aP4 = sum(sum(derivaParcialY(:,:,i).*derivaParcialY(:,:,i)));
                    aP(i,:) = [aP1 aP2 aP2 aP4];
                end

                if limpaVar == 1
                    aP1=[]; aP2=[]; aP4=[]; i=[];
                end

                % Autovalores de Ap e maior Autovalor valido
                autoVal = [;];
                autoValMax = [;];
                descAp=zeros(size(final_patches,1),1);
                ii = 1;
                for i=1:size(final_patches,1) % problemas com apagar
                    tmpAutoVal = eig(reshape(aP(i,:),[2 2]));
                    if ((tmpAutoVal(1) ~= tmpAutoVal(2)) && (isreal(tmpAutoVal(1)) && isreal(tmpAutoVal(2))))
                        autoValMax(ii) = max(tmpAutoVal(:));
                        ii = ii + 1;
                    else
                        fprintf('\n%d - %d - %f',i,ii,tmpAutoVal);
                        descAp(i)=1;
                    end
                end
                autoValMax = autoValMax';

                % Ocorreu descarte de Patch
                if sum(descAp) > 0
                    aP(descAp==1,:)=[];
                end

                if limpaVar == 1
                    autoVal=[]; tmpAutoVal=[]; i=[]; ii=[]; descAp=[]; final_patches=[];
                end

                % Determinando o Alfa
                valorAlfa = zeros(size(autoValMax,1),1);
                aCosAlfa = zeros(size(autoValMax,1),1);
                bSenAlfa = zeros(size(autoValMax,1),1);
                for i=1:size(autoValMax,1)
                    if (aP(i,2) == 0)
                        if (aP(i,1) > aP(i,4))
                            valorAlfa(i) = pi;
                        else
                            valorAlfa(i) = pi/2;
                        end
                    else
                        tmpValorAlfa = atan((autoValMax(i) - aP(i,1))/aP(i,2));
                        if (tmpValorAlfa >= -pi/2 && tmpValorAlfa < pi/4)
                            valorAlfa(i) = (pi + tmpValorAlfa);
                        else
                            valorAlfa(i) = tmpValorAlfa;
                        end
                    end
                    aCosAlfa(i) = cos(valorAlfa(i));
                    bSenAlfa(i) = sin(valorAlfa(i));
                end

                if limpaVar == 1
                    tmpValorAlfa=[]; i=[]; aP=[]; autoValMax=[];
                end

                derivaParcialU2X = zeros(patchsize,patchsize,size(valorAlfa,1));
                derivaParcialU2Y = zeros(patchsize,patchsize,size(valorAlfa,1));
                tic
                for k=1:size(valorAlfa,1)
                    for r=1:patchsize
                        for t=1:patchsize
                            tmpDeltaU2rt = [((4*aCosAlfa(k)*bSenAlfa(k))/(patchsize^3)+...
                                (4*aCosAlfa(k)*bSenAlfa(k))/(patchsize^4)-...
                                (4*aCosAlfa(k)^2)/(patchsize^3)-...
                                (4*aCosAlfa(k)^2)/(patchsize^4)-...
                                (8*aCosAlfa(k)*bSenAlfa(k)*r)/(patchsize^4)+...
                                (8*aCosAlfa(k)^2*t)/(patchsize^4))

                                ((8*aCosAlfa(k)*bSenAlfa(k)*t)/(patchsize^4)-...
                                (4*aCosAlfa(k)*bSenAlfa(k))/(patchsize^3)-...
                                (4*aCosAlfa(k)*bSenAlfa(k))/(patchsize^4)+...
                                (4*bSenAlfa(k)^2)/(patchsize^3)+...
                                (4*bSenAlfa(k)^2)/(patchsize^4)-...
                                (8*bSenAlfa(k)^2*r)/(patchsize^4))];
                            derivaParcialU2X(r,t,k) = tmpDeltaU2rt(1);
                            derivaParcialU2Y(r,t,k) = tmpDeltaU2rt(2);
                        end
                    end
                end

                if limpaVar == 1
                    tmpDeltaU2rt=[]; tmpHessU2ij=[]; k=[]; r=[]; t=[]; i=[]; j=[]; hessU2ij=[];
                    Hu2ab2ij11=[]; Hu2ab2ij12=[]; Hu2ab2ij22=[]; Huab2ij11=[]; Huab2ij12=[]; Huab2ij22=[];
                    deltau2ab2xrt=[]; deltau2ab2yrt=[]; deltauab2xrt=[]; deltauab2yrt=[];
                    u2ab2xij=[]; u2ab2yij=[]; u2intdef=[]; uab2xij=[]; uab2yij=[]; uintdef=[];
                    x=[]; y=[]; is=[]; js=[]; r=[]; t=[]; a=[]; b=[]; n=[]; u=[];
                end

                % c* e d*
                % fiCeD*
                IpUd = NaN(size(valorAlfa,1),1);
                IpU2d = NaN(size(valorAlfa,1),1);
                fiCeDAsterisco = NaN(size(valorAlfa,1),1);
                for ks=1:size(valorAlfa,1)
                    IpUd(ks)=4*aCosAlfa(ks)/patchsize^3*sum(sum(derivaParcialX(:,:,ks)))+4*bSenAlfa(ks)/(patchsize^3)*sum(sum(derivaParcialY(:,:,ks)));
                    IpU2d(ks)=sum(sum(derivaParcialX(:,:,ks).*derivaParcialU2X(:,:,ks))) + sum(sum(derivaParcialY(:,:,ks).*derivaParcialU2Y(:,:,ks)));
                    fiCeDAsterisco(ks) = sqrt(2*(1-sqrt(IpUd(ks)^2 + 3*IpU2d(ks)^2)));
                end

                if limpaVar == 1
                    ks=[]; tempoIpU2d_py=[]; tempoIpU2d_px=[];
                end

                cAsterisco = NaN(size(valorAlfa,1),1);
                dAsterisco = NaN(size(valorAlfa,1),1);
                for is=1:size(valorAlfa,1)
                    cAsterisco(is) = IpUd(is) / sqrt(IpUd(is)^2 + 3*IpU2d(is)^2);
                    dAsterisco(is) = (sqrt(3)*IpU2d(is)) / sqrt(IpUd(is)^2 + 3*IpU2d(is)^2);
                end

                if limpaVar == 1
                    is=[]; IpUd=[]; IpU2d=[];
                end

                % Determinar o Theta
                thetaP = NaN(size(dAsterisco,1),1);
                for ii=1:size(dAsterisco,1)
                    if (cAsterisco(ii) >= 0)
                        TthetaP = asin(dAsterisco(ii));
                    elseif (dAsterisco(ii) >= 0)
                        TthetaP = acos(cAsterisco(ii));
                    else
                        TthetaP = atan(abs(dAsterisco(ii))/abs(cAsterisco(ii)))+pi;
                    end
                    if ~isreal(TthetaP)
                        TthetaP = real(TthetaP);
                    end
                    thetaP(ii) = TthetaP;
                end

                if limpaVar == 1
                    ii=[]; TthetaP=[];
                end

                projetaPatch=[;];
                projetaPatchDescarta=[;];
                projetaPatchDescarta_fiCeD=[;];
                projetaPatch_fiCeD=[;];
                iz = 1;
                izd = 1;
                contii=1;
                for ii=1:size(valorAlfa,1)
                    if (fiCeDAsterisco(ii) <= distMaxima(patchsizeNumero))
                        projetaPatch(iz,:) = [valorAlfa(ii) thetaP(ii)];
                        projetaPatch_fiCeD(iz,:) = fiCeDAsterisco(ii);
                        iz = iz + 1;
                    else
                        projetaPatchDescarta(izd,:) = [valorAlfa(ii) thetaP(ii)];
                        projetaPatchDescarta_fiCeD(izd,:) = fiCeDAsterisco(ii);
                        izd = izd + 1;
                    end
                end
                
                % Se nao projetar nada
                if size(projetaPatch,1)==0
                    DescartouTodos = 'S';
                    projetaPatchDescarta_fiCeD(:,2) = (1:size(projetaPatchDescarta_fiCeD,1));
                    tmp_Descartados = sortrows(projetaPatchDescarta_fiCeD,1);
                    projetaPatch(1,:) = projetaPatchDescarta(tmp_Descartados(1,2),:);
                    projetaPatch_fiCeD(1) = tmp_Descartados(1,1);
                end

                fprintf('\nImagem: %d - Patchs: %d - Patchs Descartados: %d - Patchs Projetados: %d de %d\n',imgBase,totalPatch,izd-1,iz-1,((izd-1)+(iz-1)));
                tocPatch = toc(ticPatch);
                disp(strcat('Tempo Patch: ',datestr(datenum(0,0,0,0,0,tocPatch),'HH:MM:SS.FFF')));

                tmpPatchsFinais = [{projetaPatch} {projetaPatchDescarta} {tocPatch} {projetaPatch_fiCeD} {projetaPatchDescarta_fiCeD} {DescartouTodos}];
                PatchsFinais(patchsizeNumero,:) = tmpPatchsFinais;

                if limpaVar == 1
                    tt=[]; iz=[]; izd=[]; ti=[]; tj=[]; tTmp=[]; thetaP=[]; ks=[];
                    cAsterisco=[]; dAsterisco=[]; IpUd=[]; IpU2d=[]; aCosAlfa=[]; bSenAlfa=[]; h=[]; ii=[];
                    novo_res_patch=[]; posicao=[]; tempoIpU2d=[]; tempoIpUd=[]; vet_norma=[]; w=[];
                    deltaIp=[]; deltaPrt=[]; deltaU2rt=[]; derivaParcialU2X=[]; derivaParcialU2Y=[]; derivaParcialX=[]; derivaParcialY=[];
                    projetaPatch=[]; projetaPatchDescarta=[]; projetaPatch_fiCeD=[]; valorAlfa=[];
                    fiCeDAsterisco=[];totalPatch=[];tmpPatchsFinais=[];
                    
                end
            end
            warning('on','all')

            tocTotal = toc(ticTotal);
            disp(strcat('Imagem:',num2str(imgBase),' - Tempo Imagem: ',datestr(datenum(0,0,0,0,0,tocTotal),'HH:MM:SS.FFF')));

            if salvar == 1
                arqSave = strcat(pwd,'/tmp/','Projeta-',sprintf('%04d',imgBase),'-',num2str(cl(1)), sprintf('%02d',cl(2)), sprintf('%02d',cl(3)),'-',sprintf('%02d',cl(4)),'-',sprintf('%02d',cl(5)),'-',sprintf('%02d',cl(6)),'.mat');
                parsave(arqSave,'PatchsFinais',PatchsFinais,'matPatchsize',matPatchsize,'tocTotal',tocTotal);
            end

            if limpaVar == 1
                PatchsFinais=cell(size(matPatchsize,2),6);
            end

        end
        
        dMat = dir([pwd '/tmp/*.mat']);
        tamanhoM = length(dMat);
        PatchsFinaisTodos=cell(tamanhoM,1);
        for d = 1:tamanhoM
            fprintf('%d/%d\n',d,tamanhoM);
            load([dMat(d,1).folder '/' dMat(d,1).name]);
            PatchsFinaisTodos(d) = {PatchsFinais};
            clear('PatchsFinais','tocTotal');
        end

        if limpaVar == 1
            delete(strcat(pwd,'/tmp/*.mat'));
        end

        % Remove as celulas não utilizadas para projecao
        % Acontece quando se testa o codigo para apenas uma imagem do Banco
        PatchsFinaisTodos=PatchsFinaisTodos(~cellfun('isempty',PatchsFinaisTodos));

        tocTudo = toc(ticTudo);
        disp(strcat('Tempo Base: ',datestr(datenum(0,0,0,0,0,tocTudo),'HH:MM:SS.FFF')));

        if salvar == 1
            if ~exist('cl')
                cl = fix(clock);
            end
            arqSave = strcat(pwd,'/Resultados/Projeta-Todos-',baseN,'-',num2str(cl(1)), sprintf('%02d',cl(2)), sprintf('%02d',cl(3)),'-',sprintf('%02d',cl(4)),'-',sprintf('%02d',cl(5)),'-',sprintf('%02d',cl(6)),'.mat');
            save(arqSave,'PatchsFinaisTodos','dCell','matPatchsize','tocTudo','MatLabelS');
        end


    %
    % Calculando os Estimadores
    %

        % Frequencia de Corte (w)
        for freqCorte = FreqCorteIni:FreqCorteFim
            %>>>> Calculo do K-Fourier
            contKF = 1;
            for imgBase=1:size(PatchsFinaisTodos,1)
                fprintf('\nCalculo do K-Fourier... (%d) %s - Imagem: %d',freqCorte,baseN,imgBase);

                aChapM = zeros(freqCorte,size(matPatchsize,2));
                bChapn = zeros(floor(freqCorte/2),size(matPatchsize,2));
                cChapn = zeros(floor(freqCorte/2),size(matPatchsize,2));
                dChapmn = NaN(freqCorte-1,freqCorte-1,size(matPatchsize,2));
                eChapmn = NaN(freqCorte-1,freqCorte-1,size(matPatchsize,2));

            %   TodosProjetaPatch Alfa(1) e Theta(2)
                for iTam=1:size(matPatchsize,2)
                    tmpPatch = PatchsFinaisTodos{imgBase}{iTam,1};
                    eNeG = size(tmpPatch,1);
                    for eMe=1:freqCorte
                        tTmpA=[;];
                        for soma=1:eNeG
                            tTmpA(soma) = cos(eMe * tmpPatch(soma,2) - ((1 - ((-1)^eMe))*pi/4));
                        end
                        aChapM(eMe,iTam) = ((sqrt(2)/eNeG) * sum(tTmpA(:)));
                    end
                    if freqCorte > 1
                        for eNe=1:floor(freqCorte/2)
                            tTmpB=[;];
                            tTmpC=[;];
                            for soma=1:eNeG
                                tTmpB(soma) = cos(2 * eNe * tmpPatch(soma,1));
                                tTmpC(soma) = sin(2 * eNe * tmpPatch(soma,1));
                            end
                            bChapn(eNe,iTam) = ((sqrt(2)/eNeG) * sum(tTmpB(:)));
                            cChapn(eNe,iTam) = ((sqrt(2)/eNeG) * sum(tTmpC(:)));
                        end
                        for eNe=1:(freqCorte-1)
                            for eMe=1:(freqCorte-1)
                                if ((eNe + eMe) < (freqCorte+1))
                                    tTmpD=[;];
                                    tTmpE=[;];
                                    for soma=1:eNeG
                                        tTmpD(soma) = cos(eNe * tmpPatch(soma,1)) * cos(eMe * tmpPatch(soma,2) - ((1 - (-1)^(eMe+eNe))*pi/4));
                                        tTmpE(soma) = sin(eNe * tmpPatch(soma,1)) * cos(eMe * tmpPatch(soma,2) - ((1 - (-1)^(eMe+eNe))*pi/4));
                                    end
                                    dChapmn(eNe,eMe,iTam) = ((2/eNeG) * sum(tTmpD(:)));
                                    eChapmn(eNe,eMe,iTam) = ((2/eNeG) * sum(tTmpE(:)));
                                end
                            end
                        end
                    end

                    tmpKFourier = aChapM(1,iTam);
                    if freqCorte > 1

                        for iFreq=2:freqCorte
                            % Gera sequencia para comparacao
                            MatSeq=[;];
                            for k=1:iFreq-1
                                MatSeq = [MatSeq k*ones(1,iFreq-1)];
                            end
                            MatSeq = MatSeq';
                            tmpSeq = repmat((1:iFreq-1)',iFreq-1);
                            MatSeq(:,2) = tmpSeq(:,1);

                            % aChapM
                            tmpKFourier = [tmpKFourier aChapM(iFreq,iTam)];
                            if mod(iFreq,2) == 0
                                % bChapn e cChapn
                                tmpKFourier = [tmpKFourier bChapn(floor(iFreq/2),iTam) cChapn(floor(iFreq/2),iTam)];
                            end
                            % dChapmn
                            tmpdChapmn=[;];
                            for ikz=1:size(MatSeq,1)
                                if ~isnan(dChapmn(MatSeq(ikz,1),MatSeq(ikz,2),iTam)) && ((MatSeq(ikz,1) + MatSeq(ikz,2)) == iFreq)
                                    tmpdChapmn = [tmpdChapmn dChapmn(MatSeq(ikz,1),MatSeq(ikz,2),iTam)];
                                end
                            end
                            % eChapmn
                            tmpeChapmn=[;];
                            for ikz=1:size(MatSeq,1)
                                if ~isnan(eChapmn(MatSeq(ikz,1),MatSeq(ikz,2),iTam)) && ((MatSeq(ikz,1) + MatSeq(ikz,2)) == iFreq)
                                    tmpeChapmn = [tmpeChapmn eChapmn(MatSeq(ikz,1),MatSeq(ikz,2),iTam)];
                                end
                            end
                            tmpKFourier = [tmpKFourier tmpdChapmn tmpeChapmn];
                        end
                    end

                    clear('MatSeq','iFreq','ikz','tmpdChapmn','tmpeChapmn','tmpSeq');

                    kFourierTodos(contKF,:) = tmpKFourier;
                    juntaChap(contKF,:) = [{aChapM(:,iTam)} {bChapn(:,iTam)} {cChapn(:,iTam)} {dChapmn(:,:,iTam)} {eChapmn(:,:,iTam)}];
                    contKF = contKF + 1;
                end
            end

    %         if salvar == 1
    %             if ~exist('cl','var')
    %                 cl = fix(clock);
    %             end fprintf('\nSalvando dados... %s ',baseN); arqSave =
    %             strcat(pwd,'/tmp/KDescritor-',baseN,'-Corte-',sprintf('%02d',freqCorte),'-',num2str(cl(1)),
    %             sprintf('%02d',cl(2)),
    %             sprintf('%02d',cl(3)),'-',sprintf('%02d',cl(4)),'-',sprintf('%02d',cl(5)),'-',sprintf('%02d',cl(6)),'.mat');
    %             save(arqSave,'kFourierTodos','juntaChap','dCell','MatLabelS','matPatchsize');
    %             fprintf(' - salvos!!!');
    %         end
            
            fprintf('\nIniciando EKFC...\n');

            contKF = 1;
            contImg = 1;
            for ik=1:size(kFourierTodos,1)
                aChapM  = juntaChap{ik,1};
                bChapn  = juntaChap{ik,2};
                cChapn  = juntaChap{ik,3};
                dChapmn = juntaChap{ik,4};
                eChapmn = juntaChap{ik,5};

                if ((dChapmn(1,1)>0) && (abs(eChapmn(1,1))==0))
                    sigma = 0;
                elseif ((dChapmn(1,1)<0) && (abs(eChapmn(1,1))==0))
                    sigma = pi;
                elseif ((abs(dChapmn(1,1)) == 0) && (eChapmn(1,1)>0))
                    sigma = 3*pi/2;
                elseif ((abs(dChapmn(1,1)) == 0) && (eChapmn(1,1)<0))
                    sigma = pi/2;
                elseif ((dChapmn(1,1)~=0) && (eChapmn(1,1)~=0))
                    sigma = atan(-eChapmn(1,1)/dChapmn(1,1));
                    if ((cos(sigma)*dChapmn(1,1) - sin(sigma)*eChapmn(1,1)) < 0)
                        sigma = sigma + pi;
                    end
                    if sigma < 0
                        sigma = sigma + 2*pi;
                    end
                else
                    sigma = 0;
                end

                kF = kFourierTodos(ik,:);

                aChapinv = zeros(freqCorte);
                bChapinv = zeros(floor(freqCorte/2),1);
                cChapinv = zeros(floor(freqCorte/2),1);
                dChapinv = NaN(freqCorte-1,freqCorte-1);
                eChapinv = NaN(freqCorte-1,freqCorte-1);

                aChapinv = aChapM;

                if freqCorte > 1
                for iik=1:floor(freqCorte/2)
                    bChapinv(iik)=cos(2*iik*sigma)*bChapn(iik)-sin(2*iik*sigma)*cChapn(iik);
                    cChapinv(iik)=sin(2*iik*sigma)*bChapn(iik)+cos(2*iik*sigma)*cChapn(iik);
                end

                for eNe=1:(freqCorte-1)
                    for eMe=1:(freqCorte-1)
                        if ((eNe + eMe) < (freqCorte+1))
                            dChapinv(eNe,eMe)=cos(eNe*sigma)*dChapmn(eNe,eMe)-sin(eNe*sigma)*eChapmn(eNe,eMe);
                            eChapinv(eNe,eMe)=sin(eNe*sigma)*dChapmn(eNe,eMe)+cos(eNe*sigma)*eChapmn(eNe,eMe);
                        end
                    end
                end
                end

                tmpKFourierInv = aChapinv(1);
                if freqCorte > 1

                    for iFreq=2:freqCorte
                        % Gera sequencia para comparacao
                        MatSeq=[;];
                        for k=1:iFreq-1
                            MatSeq = [MatSeq k*ones(1,iFreq-1)];
                        end
                        MatSeq = MatSeq';
                        tmpSeq = repmat((1:iFreq-1)',iFreq-1);
                        MatSeq(:,2) = tmpSeq(:,1);

                        % aChapM
                        tmpKFourierInv = [tmpKFourierInv aChapinv(iFreq)];
                        if mod(iFreq,2) == 0
                            % bChapn e cChapn
                            tmpKFourierInv = [tmpKFourierInv bChapinv(floor(iFreq/2)) cChapinv(floor(iFreq/2))];
                        end
                        % dChapmn
                        tmpdChapmn=[;];
                        for ikz=1:size(MatSeq,1)
                            if ~isnan(dChapinv(MatSeq(ikz,1),MatSeq(ikz,2))) && ((MatSeq(ikz,1) + MatSeq(ikz,2)) == iFreq)
                                tmpdChapmn = [tmpdChapmn dChapinv(MatSeq(ikz,1),MatSeq(ikz,2))];
                            end
                        end
                        % eChapmn
                        tmpeChapmn=[;];
                        for ikz=1:size(MatSeq,1)
                            if ~isnan(eChapinv(MatSeq(ikz,1),MatSeq(ikz,2))) && ((MatSeq(ikz,1) + MatSeq(ikz,2)) == iFreq)
                                tmpeChapmn = [tmpeChapmn eChapinv(MatSeq(ikz,1),MatSeq(ikz,2))];
                            end
                        end
                        tmpKFourierInv = [tmpKFourierInv tmpdChapmn tmpeChapmn];
                    end
                end
                clear('MatSeq','iFreq','ikz','tmpdChapmn','tmpeChapmn','tmpSeq');

                KFourierInvTodos(contKF,:) = tmpKFourierInv;

                qtdDesc = size(PatchsFinaisTodos{contImg}{contKF,2},1);

                % Se nao houve descarte de Patch, coloca 1 para evitar problema divisao
                if qtdDesc == 0
                qtdDesc = 1;
                end

               
                tmpEKFC(contKF,:) = [(1 - (1 / qtdDesc)) (tmpKFourierInv / qtdDesc)];
                % EKFCi - sem divisao com indice
                tmpEKFCi(contKF,:) = [(1 - (1 / qtdDesc)) tmpKFourierInv];
                % EKFCs sem divisao
                tmpEKFCs(contKF,:) = tmpKFourierInv;
                % Estimadores dos Coeficientes de KFourier - KFourier com
                % divisoes
                tmpKFourierDiv(contKF,:) = [(1 - (1 / qtdDesc)) (kFourierTodos(ik,:) / qtdDesc)];
                % KFourier sem divisao com indice
                tmpKFourierI(contKF,:) = [(1 - (1 / qtdDesc)) kFourierTodos(ik,:)];
                % KFourier
                tmpKFourier(contKF,:) = kFourierTodos(ik,:);

                if (size(matPatchsize,2) == contKF)
                    fprintf('\nImagem : %d',contImg);
                    tmpEKFC = tmpEKFC';
                    EKFC(contImg,:) = tmpEKFC(:)';
                    tmpEKFC = [;];
                    tmpEKFCi = tmpEKFCi';
                    EKFCi(contImg,:) = tmpEKFCi(:)';
                    tmpEKFCi = [;];
                    tmpEKFCs = tmpEKFCs';
                    EKFCs(contImg,:) = tmpEKFCs(:)';
                    tmpEKFCs = [;];
                    tmpKFourierDiv = tmpKFourierDiv';
                    KFourierDiv(contImg,:) = tmpKFourierDiv(:)';
                    tmpKFourierDiv = [;];
                    tmpKFourierI = tmpKFourierI';
                    KFourierI(contImg,:) = tmpKFourierI(:)';
                    tmpKFourierI = [;];
                    tmpKFourier = tmpKFourier';
                    KFourier(contImg,:) = tmpKFourier(:)';
                    tmpKFourier = [;];
                    contImg = contImg + 1;
                    contKF = 1;
                else
                    contKF = contKF + 1;
                end
    
                if limpaVar == 1
                    clear kF sigma qtdDesc aChapM bChapn cChapn dChapmn eChapmn aChapinv bChapinv cChapinv dChapinv eChapinv tmpKFourierInv iik eNe eMe
                end
            end
            
            if salvar == 1
    %            load(strcat(pwd,'/tmp/KDescritor-',baseN,'-Corte-',sprintf('%02d',freqCorte),'-',num2str(cl(1)), sprintf('%02d',cl(2)), sprintf('%02d',cl(3)),'-',sprintf('%02d',cl(4)),'-',sprintf('%02d',cl(5)),'-',sprintf('%02d',cl(6)),'.mat'));
                
                fprintf('\nSalvando dados... %s ',baseN);
                arqSave = strcat(pwd,'/Resultados/Descritor-',baseN,'-Corte-',sprintf('%02d',freqCorte),'-',num2str(cl(1)), sprintf('%02d',cl(2)), sprintf('%02d',cl(3)),'-',sprintf('%02d',cl(4)),'-',sprintf('%02d',cl(5)),'-',sprintf('%02d',cl(6)),'.mat');
                save(arqSave,'kFourierTodos','KFourierInvTodos','juntaChap','dCell','MatLabelS','matPatchsize','EKFC','EKFCi','EKFCs','KFourierDiv','KFourierI','KFourier');
                fprintf(' - salvos!!!');
                delete(strcat(pwd,'/tmp/*.mat'));
                clear kFourierTodos KFourierInvTodos juntaChap;
            end
        end
        clear kFourierTodos KFourierInvTodos juntaChap MatLabelS base baseN;
    end

    fprintf('\n\nTerminado...\n');
    tocTudoT = toc(ticTudoF);
    disp(strcat('Tempo Total de Tudo: ',datestr(datenum(0,0,0,0,0,tocTudoT),'HH:MM:SS.FFF')));

    quit;
