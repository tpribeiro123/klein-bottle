import glob
import sys,glob
import os
import warnings
import math

from time import gmtime, strftime, time
import random
import scipy
from datetime import datetime
import datetime

import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed

import numpy as np
import matplotlib 

import cv2
from PIL import Image

def distanciaMaxima(matPatchsize):

    distMaxima = np.zeros(1,np.size(matPatchsize,2))
    for i in range(np.size(matPatchsize,2)):
        distMaxima[i] = math.sqrt(2*(1-1/(2^(matPatchsize[i]-1))))
    return distMaxima 

def retPosicao(Npatch, TamPatch, Ncolunas):
    pCol = int(Ncolunas / TamPatch)
    pLin = int(Npatch / pCol)
    rAux = Npatch - (pLin * pCol)
    if rAux == 0:
        rLin = ((pLin * TamPatch) + 1) - TamPatch
        rCol = ((pCol * TamPatch) - TamPatch) + 1
    else:
        if Npatch < pCol:
            rLin = 1
        else:
            rLin = (pLin * TamPatch) + 1
        rCol = ((rAux - 1) * TamPatch) + 1
    VarOutput = [rLin, rCol]
    return VarOutput

def busca_nomes_imagens2(path_base_name, subdir):
    dirinfo = os.scandir(path_base_name)
    for entry in dirinfo:
        if entry.name.startswith('.'):
            dirinfo.remove(entry)

    if subdir:
        dirinfo = [entry for entry in dirinfo if entry.is_dir()]
        dirinfo = [entry for entry in dirinfo if entry.name not in ['.', '..']]
        numsubdir = len(dirinfo)
        conta = 1
        imagens = []
        for K in range(numsubdir):
            filetofind = '*.png'
            subdirinfo = glob.glob(os.path.join(path_base_name, dirinfo[K].name, filetofind))
            subdirinfo += glob.glob(os.path.join(path_base_name, dirinfo[K].name, filetofind.upper()))
            subdirinfo = [os.path.basename(filepath) for filepath in subdirinfo]
            subdirinfo = [{'name': filename, 'dir': dirinfo[K].name, 'class': K} for filename in subdirinfo]
            if subdirinfo:
                imagens += subdirinfo
            else:
                filetofind = '*.jpg'
                subdirinfo = glob.glob(os.path.join(path_base_name, dirinfo[K].name, filetofind))
                subdirinfo += glob.glob(os.path.join(path_base_name, dirinfo[K].name, filetofind.upper()))
                subdirinfo = [os.path.basename(filepath) for filepath in subdirinfo]
                subdirinfo = [{'name': filename, 'dir': dirinfo[K].name, 'class': K} for filename in subdirinfo]
                if subdirinfo:
                    imagens += subdirinfo
                else:
                    filetofind = '*.tif'
                    subdirinfo = glob.glob(os.path.join(path_base_name, dirinfo[K].name, filetofind))
                    subdirinfo += glob.glob(os.path.join(path_base_name, dirinfo[K].name, filetofind.upper()))
                    subdirinfo = [os.path.basename(filepath) for filepath in subdirinfo]
                    subdirinfo = [{'name': filename, 'dir': dirinfo[K].name, 'class': K} for filename in subdirinfo]
                    if subdirinfo:
                        imagens += subdirinfo
                    else:
                        filetofind = '*.tiff'
                        subdirinfo = glob.glob(os.path.join(path_base_name, dirinfo[K].name, filetofind))
                        subdirinfo += glob.glob(os.path.join(path_base_name, dirinfo[K].name, filetofind.upper()))
                        subdirinfo = [os.path.basename(filepath) for filepath in subdirinfo]
                        subdirinfo = [{'name': filename, 'dir': dirinfo[K].name, 'class': K} for filename in subdirinfo]
                        if subdirinfo:
                            imagens += subdirinfo
    else:
        dirinfo = [entry for entry in dirinfo if entry.name not in ['.', '..']]
        imagens = [{'name': entry.name} for entry in dirinfo]

    return imagens

def Garrafa_Klein_Full():
  

  warnings.filterwarnings("ignore")

  limpaVar = False
  salvar = True

  # Frequências de Corte
  # Utilizado no loop para o corte de frequências
  FreqCorteIni = 2
  FreqCorteFim = 2

  # Matriz de Tamanhos de Patchs a Projecao da Garrafa
  # 1-Artigo, 2-Fibonacci, 3-Sequencia, 4-Seq Pares, 5-Seq Impares, 6-

  matrizPatch = 1

  if matrizPatch == 1:
      # patchs do artigo
      matPatchsize = np.array([3, 7, 11, 15, 19])
  elif matrizPatch == 2:
      # patchs fibonacci
      matPatchsize = np.array([3, 5, 8, 13, 21, 34, 55])
  elif matrizPatch == 3:
      # combina 13 patchs
      matPatchsize = np.arange(3, 28, 2)
  elif matrizPatch == 4:
      # sequencia
      matPatchsize = np.arange(3, 13)
  elif matrizPatch == 5:
      # sequencia
      matPatchsize = np.array([3, 4, 5])

  distMaxima = distanciaMaxima(matPatchsize)

  # Path para execução
  zz = os.getcwd().split('/')
  idx = [i for i, item in enumerate(map(str.lower, zz)) if 'owncloud' in item]
  if len(idx) != 0:
      idx = idx[0]
      iniciaPath = ''
      for i in range(idx):
          iniciaPath += zz[i] + '/'
  else:
      print('ERRO NAO EXISTE BASE DE IMAGENS!!!')
      exit()
      #return
  del zz, idx, i

  ticTudoF = datetime.datetime.now()
  cl = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#inicio for
  for iM in range(1):
      # Bases de Imagens
      # (1)Brodatz (2)Vistex (3)Outex (4)KTH_TIPS (5)CUReT (6)UIUCTex
      # (7)Pollen (8)SIPI (9)Sport_Event (10)Swedish_Leaf
      # Artigo: 4, 5 e 6

      # Base com subdiretórios
      bSubdir = False
      TipoTreino = 0

      if iM == 0:
          baseN = 'Brodatz'
          base = baseN + '/'
          classes = 111
          imagens = 16
      elif iM == 1:
          baseN = 'Vistex'
          base = baseN + '/'
          classes = 54
          imagens = 16
      elif iM == 2:
          baseN = 'Outex'
          base = baseN + '/'
          classes = 68
          imagens = 20
      elif iM == 3:
          baseN = 'KTH_TIPS'
          base = baseN + '/'
          classes = 10
          imagens = 81
      elif iM == 4:
          baseN = 'CUReT'
          base = 'CUReT_Mod/'
          classes = 61
          imagens = 92
      elif iM == 5:
          baseN = 'UIUCTex'
          base = 'UIUCTex/'
          classes = 25
          imagens = 40
      elif iM == 6:
          baseN = 'Pollen'
          base = 'pollen_grain_data_renamed/'
          classes = 30
          imagens = 40
          bSubdir = True
          TipoTreino = 1
      elif iM == 7:
          baseN = 'SIPI'
          base = baseN + '/'
          classes = 13
          imagens = 7
      elif iM == 8:
          baseN = 'Sport_Event'
          base = 'event_img/'
          classes = 8
          imagens = 137
          bSubdir = True
          TipoTreino = 1
      elif iM == 9:
          baseN = 'Swedish_Leaf'
          base = baseN + '/'
          classes = 15
          imagens = 75
          bSubdir = True
      elif iM == 10:
        baseN = 'Swedish_Leaf'
        base = baseN + '/'
        classes = 15
        imagens = 75
        bSubdir = 1
      elif iM == 11:
        baseN = 'alot_grey2'
        base = baseN + '/'
        classes = 250
        imagens = 100
        bSubdir = 1


      path_base_name = os.path.join(iniciaPath, 'Bases_Imagens', base)

      dCell = busca_nomes_imagens2(path_base_name, bSubdir)
      dCell = dCell[0]
      tamanhod = len(dCell)

      #define Matriz de Rotulos das Classes
      MatLabelS = np.empty((0,))
      if TipoTreino == 0:
          for k in range(classes):
              MatLabelS = np.concatenate((MatLabelS, (k+1)*np.ones(imagens)))
      else:
          for k in range(len(dCell)):
              MatLabelS = np.concatenate((MatLabelS, [dCell[k]['class']]))
      k = None
      del k

      

      # Projetando na Garrafa de Klein
      
      ticTudo = time.time()

      ################## AQUI FICA A PARALELIZAÇÃO ##################
      # 
      # 
      #  import multiprocessing
      #
      #  def process_image(imgBase):
      #      # seu código aqui
      #      pass
      #
      #  if __name__ == '__main__':
      #      pool = multiprocessing.Pool()
      #      results = pool.map(process_image, range(tamanhod))
      #      pool.close()
      #      pool.join()
      # 
      #  
      ################## AQUI FICA A PARALELIZAÇÃO ##################
      
      print(f'\nBase: {baseN} - Imagem {imgBase}...')
      ticTotal = time.time()
      for patchsizeNumero in range(matPatchsize.shape[1]):
        ticPatch = time.time()

        DescartouTodos = 'N'

        patchsize = matPatchsize[0, patchsizeNumero]

        nome = dCell[imgBase]['name']
        if 'dir' in dCell[imgBase]:
            nomeAbre = os.path.join(path_base_name, dCell[imgBase]['dir'], nome)
        else:
            nomeAbre = os.path.join(path_base_name, nome)
        input = cv2.imread(nomeAbre)

        print(f'\nPatch({patchsize}x{patchsize}) - Imagem({imgBase}): {nome}')
        if limpaVar == 1:
            nome = ""
        
        h, w, Cor = input.shape

        if Cor == 3:
            input = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)

        input = input + 1

        totalPatch = (w // patchsize) * (h // patchsize)

        if totalPatch <= 5000:
            sel_patch = np.arange(totalPatch)
        else:
            sel_patch = random.sample(range(totalPatch), 5000)

        Pos_Patch_Imagem = np.empty((2, sel_patch.shape[0]))
        for i in range(sel_patch.shape[0]):
            Pos_Patch_Imagem[:, i] = retPosicao(sel_patch[i], patchsize, w)

        todos_patch = np.empty((Pos_Patch_Imagem.shape[1], patchsize**2))
        for k in range(Pos_Patch_Imagem.shape[1]):
            tmp_img = input[Pos_Patch_Imagem[0, k]:Pos_Patch_Imagem[0, k]+patchsize, Pos_Patch_Imagem[1, k]:Pos_Patch_Imagem[1, k]+patchsize]
            todos_patch[k, :] = tmp_img.flatten()

        if limpaVar == 1:
            tmp_img = None
            input = None
            Ttodos_patch = None

        res_patch = np.log(todos_patch)

        if limpaVar == 1:
            todos_patch = None
            k = None
            i = None
        

        dnorma_patchy = []
        dnorma1 = np.zeros((patchsize-1, patchsize-1, novo_res_patch.shape[0]))
        dnorma2 = np.zeros((patchsize, patchsize, novo_res_patch.shape[0]))
        dnorma3 = np.zeros((patchsize, patchsize, novo_res_patch.shape[0]))
        temp_res_patch_Todos = []
        for ks in range(novo_res_patch.shape[0]):
            temp_res_patch = np.reshape(novo_res_patch[ks,:], (patchsize, patchsize)) - 1/(patchsize)**2 * np.mean(novo_res_patch[ks,:])
            temp_res_patch_Todos.append(temp_res_patch.ravel())
            for is_ in range(patchsize-1):
                for js in range(patchsize-1):
                    dnorma1[is_, js, ks] = (temp_res_patch[is_, js] - temp_res_patch[is_+1, js])**2 + (temp_res_patch[is_, js] - temp_res_patch[is_, js+1])**2
                dnorma2[is_, patchsize-1, ks] = (temp_res_patch[is_, patchsize-1] - temp_res_patch[is_+1, patchsize-1])**2
                dnorma3[patchsize-1, js, ks] = (temp_res_patch[patchsize-1, js] - temp_res_patch[patchsize-1, js+1])**2

        for ks in range(novo_res_patch.shape[0]):
            dnorma_patchy.append(np.sqrt(np.sum(np.sum(dnorma1[:,:,ks])) + np.sum(np.sum(dnorma2[:,:,ks])) + np.sum(np.sum(dnorma3[:,:,ks]))))

        res_patchnormalizado = np.zeros((novo_res_patch.shape[0], patchsize*patchsize))
        for ks2 in range(novo_res_patch.shape[0]):
            res_patchnormalizado[ks2,:] = (1/dnorma_patchy[ks2]) * temp_res_patch_Todos[ks2]

        if limpaVar == 1:
            dnorma1 = None
            dnorma2 = None
            dnorma3 = None
            temp_res_patch = None
            dnorma_patchy = None
            ks = None
            is_ = None
            js = None
            ks2 = None
            temp_res_patch_Todos = None

        final_patches = res_patchnormalizado

        if limpaVar == 1:
            res_patchnormalizado = None
        # DeltaP(i,j)
        deltaPij = np.empty((patchsize, patchsize, final_patches.shape[0]), dtype=object)
        # HessP(i,j)
        hessPij = np.empty((patchsize, patchsize, final_patches.shape[0]), dtype=object)

        for ii in range(final_patches.shape[0]):
            tmpPatch = np.reshape(final_patches[ii,:], (patchsize, patchsize))
            tmp_deltaPij = np.empty((patchsize, patchsize), dtype=object)
            tmp_hessPij = np.empty((patchsize, patchsize), dtype=object)
            
            for i in range(patchsize):
                for j in range(patchsize):
                    if ((i==0) or (j==0) or (i==patchsize-1) or (j==patchsize-1)):
                        tmp_deltaPij[i,j] = [[0, 0]]
                        tmp_hessPij[i,j] = [[0, 0], [0, 0]]
                    else:
                        # DeltaP
                        tmp_deltaPij[i,j] = [[(tmpPatch[i,j+1] - tmpPatch[i,j-1])/2, (tmpPatch[i-1,j] - tmpPatch[i+1,j])/2]]
                        # HessP
                        tmp_hessPij[i,j] = [[(tmpPatch[i,j+1] - 2*tmpPatch[i,j] + tmpPatch[i,j-1]),
                                            (tmpPatch[i-1,j+1] - tmpPatch[i-1,j-1] + tmpPatch[i+1,j-1] - tmpPatch[i+1,j+1])/4],
                                            [(tmpPatch[i-1,j+1] - tmpPatch[i-1,j-1] + tmpPatch[i+1,j-1] - tmpPatch[i+1,j+1])/4,
                                            (tmpPatch[i+1,j] - 2*tmpPatch[i,j] + tmpPatch[i-1,j])]]
            deltaPij[:,:,ii] = tmp_deltaPij
            hessPij[:,:,ii] = tmp_hessPij
        
        if limpaVar == 1:
            tmpDeltaPrt = []
            tmpHessPij = []
            k = []
            r = []
            t = []
            i = []
            j = []
            deltaPij = []
            hessPij = []

        aP = np.empty((final_patches.shape[0], 4))
        for i in range(final_patches.shape[0]):
            aP1 = np.sum(np.sum(derivaParcialX[:, :, i] * derivaParcialX[:, :, i]))
            aP2 = np.sum(np.sum(derivaParcialX[:, :, i] * derivaParcialY[:, :, i]))
            aP4 = np.sum(np.sum(derivaParcialY[:, :, i] * derivaParcialY[:, :, i]))
            aP[i, :] = [aP1, aP2, aP2, aP4]

        if limpaVar == 1:
            aP1 = []
            aP2 = []
            aP4 = []
            i = []

        autoValMax = []
        descAp = np.zeros((final_patches.shape[0], 1))
        for i in range(final_patches.shape[0]):
            tmpAutoVal = np.linalg.eig(np.reshape(aP[i, :], [2, 2]))
            if (tmpAutoVal[0] != tmpAutoVal[1]) and np.isreal(tmpAutoVal).all():
                autoValMax.append(np.max(tmpAutoVal))
            else:
                print("\n%d - %d - %f" % (i, ii, tmpAutoVal))
                descAp[i] = 1

        if len(autoValMax) > 0:
            autoValMax = autoValMax

        if np.sum(descAp) > 0:
            aP = aP[descAp == 0, :]

        if limpaVar == 1:
            autoVal = []
            tmpAutoVal = []
            i = []
            ii = []
            descAp = []
            final_patches = []

        valorAlfa = np.zeros((len(autoValMax), 1))
        aCosAlfa = np.zeros((len(autoValMax), 1))
        bSenAlfa = np.zeros((len(autoValMax), 1))
        for i in range(len(autoValMax)):
            if aP[i, 2] == 0:
                if aP[i, 0] > aP[i, 3]:
                    valorAlfa[i] = np.pi
                else:
                    valorAlfa[i] = np.pi / 2
            else:
                tmpValorAlfa = np.arctan((autoValMax[i] - aP[i, 0]) / aP[i, 2])
                if (tmpValorAlfa >= -np.pi / 2) and (tmpValorAlfa < np.pi / 4):
                    valorAlfa[i] = (np.pi + tmpValorAlfa)
                else:
                    valorAlfa[i] = tmpValorAlfa
            aCosAlfa[i] = np.cos(valorAlfa[i])
            bSenAlfa[i] = np.sin(valorAlfa[i])

        if limpaVar == 1:
            tmpValorAlfa = []
            i = []
            aP = []
            autoValMax = []

        derivaParcialU2X = np.zeros((patchsize, patchsize, valorAlfa.shape[0]))
        derivaParcialU2Y = np.zeros((patchsize, patchsize, valorAlfa.shape[0]))
        tic = 0

        for k in range(valorAlfa.shape[0]):
            for r in range(patchsize):
                for t in range(patchsize):
                    tmpDeltaU2rt = [
                        ((4*aCosAlfa[k]*bSenAlfa[k])/(patchsize**3) +
                        (4*aCosAlfa[k]*bSenAlfa[k])/(patchsize**4) -
                        (4*aCosAlfa[k]**2)/(patchsize**3) -
                        (4*aCosAlfa[k]**2)/(patchsize**4) -
                        (8*aCosAlfa[k]*bSenAlfa[k]*r)/(patchsize**4) +
                        (8*aCosAlfa[k]**2*t)/(patchsize**4)),

                        ((8*aCosAlfa[k]*bSenAlfa[k]*t)/(patchsize**4) -
                        (4*aCosAlfa[k]*bSenAlfa[k])/(patchsize**3) -
                        (4*aCosAlfa[k]*bSenAlfa[k])/(patchsize**4) +
                        (4*bSenAlfa[k]**2)/(patchsize**3) +
                        (4*bSenAlfa[k]**2)/(patchsize**4) -
                        (8*bSenAlfa[k]**2*r)/(patchsize**4))
                    ]
                    derivaParcialU2X[r, t, k] = tmpDeltaU2rt[0]
                    derivaParcialU2Y[r, t, k] = tmpDeltaU2rt[1]

        if limpaVar == 1:
            tmpDeltaU2rt = []
            tmpHessU2ij = []
            k = []
            r = []
            t = []
            i = []
            j = []
            hessU2ij = []
            Hu2ab2ij11 = []
            Hu2ab2ij12 = []
            Hu2ab2ij22 = []
            Huab2ij11 = []
            Huab2ij12 = []
            Huab2ij22 = []
            deltau2ab2xrt = []
            deltau2ab2yrt = []
            deltauab2xrt = []
            deltauab2yrt = []
            u2ab2xij = []
            u2ab2yij = []
            u2intdef = []
            uab2xij = []
            uab2yij = []
            uintdef = []
            x = []
            y = []
            _is = []
            js = []
            r = []
            t = []
            a = []
            b = []
            n = []
            u = []

        IpUd = np.full((valorAlfa.shape[0],), np.nan)
        IpU2d = np.full((valorAlfa.shape[0],), np.nan)
        fiCeDAsterisco = np.full((valorAlfa.shape[0],), np.nan)         

        for ks in range(valorAlfa.shape[0]):
            IpUd[ks] = 4*aCosAlfa[ks]/patchsize**3*np.sum(np.sum(derivaParcialX[:,:,ks])) + 4*bSenAlfa(ks)/(patchsize**3)*np.sum(np.sum(derivaParcialY[:,:,ks]))
            IpU2d[ks] = np.sum(np.sum(derivaParcialX[:,:,ks]*derivaParcialU2X[:,:,ks])) + np.sum(np.sum(derivaParcialY[:,:,ks]*derivaParcialU2Y[:,:,ks]))
            fiCeDAsterisco[ks] = np.sqrt(2*(1-np.sqrt(IpUd(ks)**2 + 3*IpU2d(ks)**2)))

        if limpaVar == 1:
            ks = []
            tempoIpU2d_py = []
            tempoIpU2d_px = []

        cAsterisco = np.full((valorAlfa.shape[0], 1), np.nan)
        dAsterisco = np.full((valorAlfa.shape[0], 1), np.nan)

        for _is in range(valorAlfa.shape[0]):
            cAsterisco[_is] = IpUd[_is] / np.sqrt(IpUd[_is]**2 + 3*IpU2d[_is]**2)
            dAsterisco[_is] = (np.sqrt(3)*IpU2d[_is]) / np.sqrt(IpUd[_is]**2 + 3*IpU2d[_is]**2) 



        if limpaVar == 1:
            is_array, IpUd_array, IpU2d_array = [], [], []

        thetaP = np.full((dAsterisco.shape[0], 1), np.nan)
        for ii in range(dAsterisco.shape[0]):
            if cAsterisco[ii] >= 0:
                TthetaP = np.arcsin(dAsterisco[ii])
            elif dAsterisco[ii] >= 0:
                TthetaP = np.arccos(cAsterisco[ii])
            else:
                TthetaP = np.arctan(np.abs(dAsterisco[ii]) / np.abs(cAsterisco[ii])) + np.pi
            if not np.isreal(TthetaP):
                TthetaP = np.real(TthetaP)
            thetaP[ii] = TthetaP

        if limpaVar == 1:
            ii, TthetaP = [], []

        projetaPatch, projetaPatchDescarta, projetaPatchDescarta_fiCeD, projetaPatch_fiCeD = np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))
        iz, izd, contii = 1, 1, 1
        for ii in range(valorAlfa.shape[0]):
            if fiCeDAsterisco[ii] <= distMaxima(patchsizeNumero):
                projetaPatch = np.vstack([projetaPatch, [valorAlfa[ii], thetaP[ii]]])
                projetaPatch_fiCeD = np.vstack([projetaPatch_fiCeD, fiCeDAsterisco[ii]])
                iz += 1
            else:
                projetaPatchDescarta = np.vstack([projetaPatchDescarta, [valorAlfa[ii], thetaP[ii]]])
                projetaPatchDescarta_fiCeD = np.vstack([projetaPatchDescarta_fiCeD, fiCeDAsterisco[ii]])
                izd += 1

        if projetaPatch.shape[0] == 0:
            DescartouTodos = 'S'
            projetaPatchDescarta_fiCeD[:, 1] = np.arange(1, projetaPatchDescarta_fiCeD.shape[0]+1)
            tmp_Descartados = projetaPatchDescarta[np.argsort(projetaPatchDescarta_fiCeD[:, 0]), :]
            projetaPatch = np.vstack([projetaPatch, tmp_Descartados[0, :]])
            projetaPatch_fiCeD[0] = tmp_Descartados[0, 0]

        print(f"\nImagem: {imgBase} - Patchs: {totalPatch} - Patchs Descartados: {izd-1} - Patchs Projetados: {iz-1} de {izd-1+iz-1}")
        tocPatch = time.time() - ticPatch
        print(f"Tempo Patch: {datetime.timedelta(seconds=tocPatch)}")

        tmpPatchsFinais = [projetaPatch, projetaPatchDescarta, tocPatch, projetaPatch_fiCeD, projetaPatchDescarta_fiCeD, DescartouTodos]
        PatchsFinais[patchsizeNumero, :] = tmpPatchsFinais
                      
        if limpaVar == 1:
            tt = []; iz = []; izd = []; ti = []; tj = []; tTmp = []; thetaP = []; ks = [];
            cAsterisco = []; dAsterisco = []; IpUd = []; IpU2d = []; aCosAlfa = []; bSenAlfa = []; h = []; ii = [];
            novo_res_patch = []; posicao = []; tempoIpU2d = []; tempoIpUd = []; vet_norma = []; w = [];
            deltaIp = []; deltaPrt = []; deltaU2rt = []; derivaParcialU2X = []; derivaParcialU2Y = []; derivaParcialX = []; derivaParcialY = [];
            projetaPatch = []; projetaPatchDescarta = []; projetaPatch_fiCeD = []; valorAlfa = [];
            fiCeDAsterisco = []; totalPatch = []; tmpPatchsFinais = [];

        os.warning('on','all')

        tocTotal = time.monotonic() - ticTotal
        print('Imagem:', str(imgBase), '- Tempo Imagem: ', time.strftime('%H:%M:%S.%f', time.gmtime(tocTotal)))

        if salvar == 1:
            arqSave = f"{os.getcwd()}/tmp/Projeta-{imgBase:04d}-{cl[0]}{cl[1]:02d}{cl[2]:02d}-{cl[3]:02d}-{cl[4]:02d}-{cl[5]:02d}.mat"
            data = {
                "PatchsFinais": PatchsFinais,
                "matPatchsize": matPatchsize,
                "tocTotal": tocTotal,
            }
            scipy.io.savemat(arqSave, data)



        if limpaVar == 1:
            PatchsFinais = [[[] for i in range(6)] for j in range(len(matPatchsize))]


        dMat = os.listdir('tmp')
        tamanhoM = len(dMat)
        PatchsFinaisTodos = [None]*tamanhoM
        for d in range(tamanhoM):
            print(f'{d+1}/{tamanhoM}')
            mat = np.load(f'tmp/{dMat[d]}', allow_pickle=True)
            PatchsFinaisTodos[d] = mat['PatchsFinais']
            del mat['PatchsFinais']
            del mat['tocTotal']

        if limpaVar == 1:
            os.system('rm -f tmp/*.mat')

        PatchsFinaisTodos = [i for i in PatchsFinaisTodos if i is not None]

        tocTudo = time.time() - ticTudo
        print(f'Tempo Base: {datetime.timedelta(seconds=tocTudo)}')

        if salvar == 1:
            if not 'cl' in locals():
                cl = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            arqSave = f"Resultados/Projeta-Todos-{baseN}-{cl}.mat"
            np.savez(arqSave, PatchsFinaisTodos=PatchsFinaisTodos, dCell=dCell, matPatchsize=matPatchsize, tocTudo=tocTudo, MatLabelS=MatLabelS)
        

        for freqCorte in range(FreqCorteIni, FreqCorteFim + 1):
            contKF = 1
            for imgBase in range(PatchsFinaisTodos.shape[0]):
                print('\nCalculo do K-Fourier... ({}) {} - Imagem: {}'.format(freqCorte, baseN, imgBase+1))

                aChapM = np.zeros((freqCorte, matPatchsize.shape[1]))
                bChapn = np.zeros((freqCorte//2, matPatchsize.shape[1]))
                cChapn = np.zeros((freqCorte//2, matPatchsize.shape[1]))
                dChapmn = np.empty((freqCorte-1, freqCorte-1, matPatchsize.shape[1]))
                eChapmn = np.empty((freqCorte-1, freqCorte-1, matPatchsize.shape[1]))

                for iTam in range(matPatchsize.shape[1]):
                    tmpPatch = PatchsFinaisTodos[imgBase][iTam, 0]
                    eNeG = tmpPatch.shape[0]
                    for eMe in range(1, freqCorte+1):
                        tTmpA = np.empty((eNeG,))
                        for soma in range(eNeG):
                            tTmpA[soma] = np.cos(eMe * tmpPatch[soma, 1] - ((1 - ((-1)**eMe))*np.pi/4))
                        aChapM[eMe-1, iTam] = ((np.sqrt(2)/eNeG) * np.sum(tTmpA))

                    if freqCorte > 1:
                        for eNe in range(1, freqCorte//2+1):
                            tTmpB = np.empty((eNeG,))
                            tTmpC = np.empty((eNeG,))
                            for soma in range(eNeG):
                                tTmpB[soma] = np.cos(2 * eNe * tmpPatch[soma, 0])
                                tTmpC[soma] = np.sin(2 * eNe * tmpPatch[soma, 0])
                            bChapn[eNe-1, iTam] = ((np.sqrt(2)/eNeG) * np.sum(tTmpB))
                            cChapn[eNe-1, iTam] = ((np.sqrt(2)/eNeG) * np.sum(tTmpC))

                        for eNe in range(1, freqCorte):
                            for eMe in range(1, freqCorte):
                                if ((eNe + eMe) < (freqCorte+1)):
                                    tTmpD = np.empty((eNeG,))
                                    tTmpE = np.empty((eNeG,))
                                    for soma in range(eNeG):
                                        tTmpD[soma] = np.cos(eNe * tmpPatch[soma, 0]) * np.cos(eMe * tmpPatch[soma, 1] - ((1 - (-1)**(eMe+eNe))*np.pi/4))
                                        tTmpE[soma] = np.sin(eNe * tmpPatch[soma, 0]) * np.cos(eMe * tmpPatch[soma, 1] - ((1 - (-1)**(eMe+eNe))*np.pi/4))
                                    dChapmn[eNe-1, eMe-1, iTam] = ((2/eNeG) * np.sum(tTmpD))
                                    eChapmn[eNe-1, eMe-1, iTam] = ((2/eNeG) * np.sum(tTmpE))
                                   
                                    tmpKFourier = aChapM[0, iTam]
                    if freqCorte > 1:
                        for iFreq in range(2, freqCorte+1):
                            MatSeq = []
                            for k in range(1, iFreq):
                                MatSeq += [k]*(iFreq-1)
                                MatSeq = np.array(MatSeq).reshape(iFreq-1, iFreq-1)
                                tmpSeq = np.tile(np.arange(1, iFreq), (iFreq-1, 1))
                                MatSeq[:, 1] = tmpSeq[:, 0]
                                tmpKFourier = aChapM[0, iTam]
                    if freqCorte > 1:
                        for iFreq in range(2, freqCorte+1):
                            MatSeq = []
                            for k in range(1, iFreq):
                                MatSeq += [k]*(iFreq-1)
                                MatSeq = np.array(MatSeq).reshape(iFreq-1, iFreq-1)
                                tmpSeq = np.tile(np.arange(1, iFreq), (iFreq-1, 1))
                                MatSeq[:, 1] = tmpSeq[:, 0]

                                tmpKFourier = np.hstack([tmpKFourier, aChapM[iFreq-1, iTam]])
                                if iFreq % 2 == 0:
                                    tmpKFourier = np.hstack([tmpKFourier, bChapn[iFreq//2-1, iTam], cChapn[iFreq//2-1, iTam]])
                                tmpdChapmn = []
                                for ikz in range(MatSeq.shape[0]):
                                    if not np.isnan(dChapmn[MatSeq[ikz, 0]-1, MatSeq[ikz, 1]-1, iTam]) and ((MatSeq[ikz, 0] + MatSeq[ikz, 1]) == iFreq):
                                        tmpdChapmn.append(dChapmn[MatSeq[ikz, 0]-1, MatSeq[ikz, 1]-1, iTam])
                                tmpeChapmn = []
                                for ikz in range(MatSeq.shape[0]):
                                    if not np.isnan(eChapmn[MatSeq[ikz, 0]-1, MatSeq[ikz, 1]-1, iTam]) and ((MatSeq[ikz, 0] + MatSeq[ikz, 1]) == iFreq):
                                        tmpeChapmn.append(eChapmn[MatSeq[ikz, 0]-1, MatSeq[ikz, 1]-1, iTam])
                                tmpKFourier = np.hstack([tmpKFourier, tmpdChapmn, tmpeChapmn])

                    del MatSeq, iFreq, ikz, tmpdChapmn, tmpeChapmn, tmpSeq
                    kFourierTodos[contKF,:] = tmpKFourier
                    juntaChap[contKF,:] = [aChapM[:,iTam], bChapn[:,iTam], cChapn[:,iTam], dChapmn[:,:,iTam], eChapmn[:,:,iTam]]
                    contKF += 1

                    # if salvar == 1
                    #     if not 'cl' in locals():
                    #         cl = np.fix(time.localtime())
                    #     print('\nSalvando dados... {} '.format(baseN))
                    #     arqSave = '{}/tmp/KDescritor-{}-Corte-{}-{}{:02d}{:02d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}.mat'.format(
                    #         os.getcwd(), baseN, freqCorte, cl.tm_year, cl.tm_mon, cl.tm_mday, cl.tm_hour, cl.tm_min, cl.tm_sec, 0)
                    #     np.savez(arqSave, kFourierTodos=kFourierTodos, juntaChap=juntaChap, dCell=dCell, MatLabelS=MatLabelS, matPatchsize=matPatchsize)
                    #     print(' - salvos!!!')

                    

                print("\nIniciando EKFC...\n")

                contKF = 1
                contImg = 1
                for ik in range(kFourierTodos.shape[0]):
                    aChapM = juntaChap[ik][0]
                    bChapn = juntaChap[ik][1]
                    cChapn = juntaChap[ik][2]
                    dChapmn = juntaChap[ik][3]
                    eChapmn = juntaChap[ik][4]

                    if ((dChapmn[0,0]>0) and (abs(eChapmn[0,0])==0)):
                        sigma = 0
                    elif ((dChapmn[0,0]<0) and (abs(eChapmn[0,0])==0)):
                        sigma = np.pi
                    elif ((abs(dChapmn[0,0]) == 0) and (eChapmn[0,0]>0)):
                        sigma = 3*np.pi/2
                    elif ((abs(dChapmn[0,0]) == 0) and (eChapmn[0,0]<0)):
                        sigma = np.pi/2
                    elif ((dChapmn[0,0]!=0) and (eChapmn[0,0]!=0)):
                        sigma = np.arctan(-eChapmn[0,0]/dChapmn[0,0])
                        if ((np.cos(sigma)*dChapmn[0,0] - np.sin(sigma)*eChapmn[0,0]) < 0):
                            sigma = sigma + np.pi
                        if sigma < 0:
                            sigma = sigma + 2*np.pi
                    else:
                        sigma = 0

                    kF = kFourierTodos[ik,:]

                    aChapinv = np.zeros(freqCorte)
                    bChapinv = np.zeros(np.floor(freqCorte/2), dtype=int)
                    cChapinv = np.zeros(np.floor(freqCorte/2), dtype=int)
                    dChapinv = np.full((freqCorte-1, freqCorte-1), np.nan)
                    eChapinv = np.full((freqCorte-1, freqCorte-1), np.nan)

                    aChapinv = aChapM

                    
                    MatSeq = None
                    iFreq = None
                    ikz = None
                    tmpdChapmn = None
                    tmpeChapmn = None
                    tmpSeq = None

                    KFourierInvTodos[contKF, :] = tmpKFourierInv

                    qtdDesc = PatchsFinaisTodos[contImg][contKF, 1].shape[0]

                    if qtdDesc == 0:
                        qtdDesc = 1

                    tmpEKFC[contKF, :] = [(1 - (1 / qtdDesc)), (tmpKFourierInv / qtdDesc)]
                    tmpEKFCi[contKF, :] = [(1 - (1 / qtdDesc)), tmpKFourierInv]
                    tmpEKFCs[contKF, :] = tmpKFourierInv
                    tmpKFourierDiv[contKF, :] = [(1 - (1 / qtdDesc)), (kFourierTodos[ik, :] / qtdDesc)]
                    tmpKFourierI[contKF, :] = [(1 - (1 / qtdDesc)), kFourierTodos[ik, :]]
                    tmpKFourier[contKF, :] = kFourierTodos[ik, :]
                    
                    EKFC = []
                    EKFCi = []
                    EKFCs = []
                    KFourierDiv = []
                    KFourierI = []
                    KFourier = []

                    if matPatchsize.shape[1] == contKF:
                        print('\nImagem :', contImg)
                        tmpEKFC = tmpEKFC
                        EKFC[contImg, :] = tmpEKFC.flatten()
                        tmpEKFC = np.empty((0, 0))
                        tmpEKFCi = tmpEKFCi
                        EKFCi[contImg, :] = tmpEKFCi.flatten()
                        tmpEKFCi = np.empty((0, 0))
                        tmpEKFCs = tmpEKFCs
                        EKFCs[contImg, :] = tmpEKFCs.flatten()
                        tmpEKFCs = np.empty((0, 0))
                        tmpKFourierDiv = tmpKFourierDiv
                        KFourierDiv[contImg, :] = tmpKFourierDiv.flatten()
                        tmpKFourierDiv = np.empty((0, 0))
                        tmpKFourierI = tmpKFourierI
                        KFourierI[contImg, :] = tmpKFourierI.flatten()
                        tmpKFourierI = np.empty((0, 0))
                        tmpKFourier = tmpKFourier
                        KFourier[contImg, :] = tmpKFourier.flatten()
                        tmpKFourier = np.empty((0, 0))
                        contImg += 1
                        contKF = 1
                    else:
                        contKF += 1

                    if limpaVar == 1:
                        del kF, sigma, qtdDesc, aChapM, bChapn, cChapn, dChapmn, eChapmn, aChapinv, bChapinv, cChapinv, dChapinv, eChapinv, tmpKFourierInv, iik, eNe, eMe

                

            if salvar == 1:
                print('\nSalvando dados... {}'.format(baseN))
                arqSave = '{}/Resultados/Descritor-{}-Corte-{}-{}{}{}-{}-{}-{}.mat'.format(os.getcwd(), baseN, '{:02d}'.format(freqCorte), cl[0], '{:02d}'.format(cl[1]), '{:02d}'.format(cl[2]), '{:02d}'.format(cl[3]), '{:02d}'.format(cl[4]), '{:02d}'.format(cl[5]))
                scipy.io.savemat(arqSave, {'kFourierTodos': kFourierTodos, 'KFourierInvTodos': KFourierInvTodos, 'juntaChap': juntaChap, 'dCell': dCell, 'MatLabelS': MatLabelS, 'matPatchsize': matPatchsize, 'EKFC': EKFC, 'EKFCi': EKFCi, 'EKFCs': EKFCs, 'KFourierDiv': KFourierDiv, 'KFourierI': KFourierI, 'KFourier': KFourier})
                print(' - salvos!!!')
                for file in glob.glob(os.path.join(os.getcwd(), 'tmp', '*.mat')):
                    os.remove(file)
                kFourierTodos = None
                KFourierInvTodos = None
                juntaChap = None
                del kFourierTodos, KFourierInvTodos, juntaChap
            MatLabelS = None
            base = None
            baseN = None
            kFourierTodos = None
            KFourierInvTodos = None
            juntaChap = None
            
        del MatLabelS, base, baseN, kFourierTodos, KFourierInvTodos, juntaChap

      print("\n\nTerminado...\n")
      tocTudoT = time.time() - ticTudoF
      print('Tempo Total de Tudo: {}'.format(time.strftime('%H:%M:%S.%f', time.gmtime(tocTudoT))))
      sys.exit()          

def main():
  Garrafa_Klein_Full()

if __name__ == '__main__':
    main()