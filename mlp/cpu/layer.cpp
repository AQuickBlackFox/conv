#include<iostream>
#include<thread>
#include<cmath>

template<typename T, int height, int width>
struct Matrix{
public:
  float *ptr;
  Matrix(){
    ptr = new float[height*width];
  }
  ~Matrix(){
    delete ptr;
  }
  T index(int x, int y){
    return ptr[x + y*width];
  }
  void setAtIndex(int x, int y, T val){
    ptr[x+ y*width] = val;
  }
  void set(T val){
    for(int h=0;h<height;h++){
      for(int w=0;w<width;w++){
        ptr[w+h*width] = val;
      }
    }
  }
};

template<typename T, int S_HEIGHT, int S_WIDTH>
void Tanh(Matrix<T, S_HEIGHT, S_WIDTH> &S, Matrix<T, S_HEIGHT, S_WIDTH> &H){
  for(int h=0;h<S_HEIGHT;h++){
    for(int w=0;w<S_WIDTH;w++){
      T a = S.index(w, h);
      T den = (std::exp(a) + std::exp(-a));
      T num = (std::exp(a) - std::exp(-a));
      T val = num / den;
      H.setAtIndex(w, h, val);
    }
  }
}

template<typename T, int W_HEIGHT, int W_WIDTH, int X_WIDTH>
void Dot(Matrix<T, W_HEIGHT, X_WIDTH> &S, Matrix<T, W_HEIGHT, W_WIDTH> &W, Matrix<T, W_WIDTH, X_WIDTH> &X){
  for(int wh=0;wh<W_HEIGHT;wh++){
    for(int xw=0;xw<X_WIDTH;xw++){
      T f = 0;
      for(int ww=0;ww<W_WIDTH;ww++){
        f = f + W.index(ww, wh) * X.index(xw, ww);
      }
      S.setAtIndex(xw, wh, f);
    }
  }
}

template<typename T, int HEIGHT, int WIDTH>
void Add(Matrix<T, HEIGHT, WIDTH> &S, Matrix<T, HEIGHT, WIDTH> &A, Matrix<T, HEIGHT, WIDTH> &B){
  for(int h=0;h<HEIGHT;h++){
    for(int w=0;w<WIDTH;w++){
      S.setAtIndex(w, h, A.index(w, h) + B.index(w, h));
    }
  }
}

int main(){
  Matrix<float, 128, 128> X1;
  X1.set(1.0f);
  Matrix<float, 4, 128> W1;
  W1.set(0.1f);
  Matrix<float, 4, 128> B1;
  B1.set(0.5f);
  Matrix<float, 4, 128> S1;
  S1.set(0.0f);
  Matrix<float, 4, 128> H1;
  H1.set(0.0f);
  Matrix<float, 10, 4> W2;
  W2.set(0.2f);
  Matrix<float, 10, 128> B2;
  B2.set(0.3f);
  Matrix<float, 10, 128> S2;
  S2.set(0.0f);
  Matrix<float, 10, 128> H2;
  H2.set(0.0f);
  Dot(S1, W1, X1);
  Add(S1, S1, B1);
  Tanh(S1, H1);
  Dot(S2, W2, H1);
  Add(S2, S2, B2);
  std::cout<<S2.index(12,2)<<std::endl;
}
