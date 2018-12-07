#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define PI 3.14159265358979323846264338327

int main(int argc, char **argv){
  /* Inicializamos variables*/

  /*Para leer los parametros y convertirlos al formato que corresponde*/
  int N = 1000;
  double mu = 0.0;
  double sigma = 1.0;
    
  double normal(double x,double sigma,double mu)
  {
    double y=1/(sigma * sqrt(2 * PI)) * exp( - pow ( (x-mu)/ (sqrt(2)*sigma),2));
    return y;
  }
  
  double min(double a, double b)
  {
    if(a<b)
    {return a;}
    else
    {return b;}
  }  
  
  void metropolis(double *lista, double *propuestas, double *alphas, int N)
  {
      srand48(N);
      double rand0=drand48();
      lista[0]=rand0;
      int i;
      double sig=4.0;
      #pragma omp parallel for shared(lista)
      for (i=1; i<N; i++)
      {
          double propuesta=lista[i-1]+(drand48()*sig-(sig/2));
          double r=min(1.0, normal(propuesta, mu, sigma)/normal(lista[i-1], mu, sigma));
          double alpha=drand48();
          alphas[i]=alpha;
          propuestas[i]=propuesta;
          if(alpha<r)
          {
              lista[i]=propuesta;
          }
          else{
              lista[i]=lista[i-1];
          }
      }
      
  }
  int j;  
  
  for (j=0;j<8;j++)
  {
      FILE *out;
      char filename[128];
      double *lista=malloc(sizeof(double)*N);
      double *alphas=malloc(sizeof(double)*N);
      double *propuestas=malloc(sizeof(double)*N);
      metropolis(lista, propuestas, alphas, N);
      sprintf(filename, "sample_%d.dat", j);
    
      if(!(out = fopen(filename, "w"))){
                fprintf(stderr, "Problema abriendo el archivo\n");
                exit(1);
        }
      int i;
      for(i=0;i<N;i++){
                fprintf(out, "%f\n", lista[i]);
        }

        fclose(out);
  }
  return 0;
}
