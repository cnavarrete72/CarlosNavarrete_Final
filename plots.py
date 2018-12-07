import numpy as np 
import matplotlib.pyplot as plt 

archivo_1 = np.loadtxt('sample_0.dat')
archivo_2 = np.loadtxt('sample_1.dat')
archivo_3 = np.loadtxt('sample_2.dat')
archivo_4 = np.loadtxt('sample_3.dat')
archivo_5 = np.loadtxt('sample_4.dat')
archivo_6 = np.loadtxt('sample_5.dat')
archivo_7 = np.loadtxt('sample_6.dat')
archivo_8 = np.loadtxt('sample_7.dat')

archivos = [archivo_1, archivo_2, archivo_3, archivo_4, archivo_5, archivo_6, archivo_7, archivo_8]

def normal(x,mu,sigma):
    #Esta es la fórmula de la distribución.
    y=1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (x - mu)**2 / (2 * sigma**2) )
    return y

mu = 0.0
sigma = 1.0
x = np.linspace(-4.0, 4.0, 300)

plt.figure()
for i, arch in zip(range(8), archivos): 
	plt.subplot(4, 4, i+1)
	plt.hist(arch, bins=100, normed=True)
	plt.plot(x, normal(x, mu, sigma), linewidth=2, color='r')

plt.savefig('fig_punto1.pdf')