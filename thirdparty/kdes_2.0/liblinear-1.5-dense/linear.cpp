#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "linear.h"
#include "tron.h"
typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{   
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

class l2r_lr_fun : public function
{
public:
	l2r_lr_fun(const problem *prob, double Cp, double Cn);
	~l2r_lr_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	const problem *prob;
};

l2r_lr_fun::l2r_lr_fun(const problem *prob, double Cp, double Cn)
{
	int i;
	int l=prob->l;
	int *y=prob->y;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	C = new double[l];

	for (i=0; i<l; i++)
	{
		if (y[i] == 1)
			C[i] = Cp;
		else
			C[i] = Cn;
	}
}

l2r_lr_fun::~l2r_lr_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
}


double l2r_lr_fun::fun(double *w)
{
	int i;
	double f=0;
	int *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);
	for(i=0;i<l;i++)
	{
		double yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i]*log(1 + exp(-yz));
		else
			f += C[i]*(-yz+log(1 + exp(yz)));
	}
	f = 2*f;
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

void l2r_lr_fun::grad(double *w, double *g)
{
	int i;
	int *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + g[i];
}

int l2r_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	double *wa = new double[l];

	Xv(s, wa);
	for(i=0;i<l;i++)
		wa[i] = C[i]*D[i]*wa[i];

	XTv(wa, Hs);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void l2r_lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;

#ifdef _DENSE_REP
        int w_size = get_nr_variable();
	double **x = prob->x;
	
	for(i=0;i<l;i++)
	{
		double *s=x[i];
		Xv[i]=0;
		for(int j=0;j<w_size;j++)
		{
                        Xv[i]+=v[j]*s[j];
		}
	}
#else
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
#endif
}

void l2r_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();

#ifdef _DENSE_REP
 	double **x = prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;

	for(i=0;i<l;i++)
	{
		double *s=x[i];
		for(int j=0;j<w_size;j++)
		{
		        XTv[j]+=v[i]*s[j];
		}
	}
#else
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
#endif
}

class l2r_l2_svc_fun : public function
{
public:
	l2r_l2_svc_fun(const problem *prob, double Cp, double Cn);
	~l2r_l2_svc_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void subXv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const problem *prob;
};

l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double Cp, double Cn)
{
	int i;
	int l=prob->l;
	int *y=prob->y;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	C = new double[l];
	I = new int[l];

	for (i=0; i<l; i++)
	{
		if (y[i] == 1)
			C[i] = Cp;
		else
			C[i] = Cn;
	}
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
	delete[] I;
}

double l2r_l2_svc_fun::fun(double *w)
{
	int i;
	double f=0;
	int *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}
	f = 2*f;
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	int *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

int l2r_l2_svc_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	double *wa = new double[l];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;

#ifdef _DENSE_REP
	int j,w_size = get_nr_variable();
	double **x = prob->x;


	for(i=0;i<l;i++)
	{
		double *s=x[i];
		Xv[i]=0;
		for(j=0;j<w_size;j++)
		{
			Xv[i]+=v[j]*s[j];

		}
	}

#else
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
#endif
}

void l2r_l2_svc_fun::subXv(double *v, double *Xv)
{
	int i;

#ifdef _DENSE_REP
	int j,w_size = get_nr_variable();
	double **x=prob->x;

	for(i=0;i<sizeI;i++)
	{
		double *s=x[I[i]];
		Xv[i]=0;
		for(j=0;j<w_size;j++)
		{
		        Xv[i]+=v[j]*s[j];
		}
	}
#else
	feature_node **x=prob->x;

	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
#endif
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();

#ifdef _DENSE_REP
	double **x=prob->x;
	int j;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
	{
		double *s=x[I[i]];
		for(j=0;j<w_size;j++)
		{
			XTv[j]+=v[i]*s[j];
		}
	}
	
#else
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
#endif
}

// A coordinate descent algorithm for 
// multi-class support vector machines by Crammer and Singer
//
//  min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
//    s.t.     \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
// 
//  where e^m_i = 0 if y_i  = m,
//        e^m_i = 1 if y_i != m,
//  C^m_i = C if m  = y_i, 
//  C^m_i = 0 if m != y_i, 
//  and w_m(\alpha) = \sum_i \alpha^m_i x_i 
//
// Given: 
// x, y, C
// eps is the stopping tolerance
//
// solution will be put in w
class Solver_MCSVM_CS
{
	public:
		Solver_MCSVM_CS(const problem *prob, int nr_class, double *C, double eps=0.1, int max_iter=100000);
		~Solver_MCSVM_CS();
		void Solve(double *w);
	private:
		void solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new);
		bool be_shrunk(int m, int yi, double alpha_i, double minG);
		double *B, *C, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};

Solver_MCSVM_CS::Solver_MCSVM_CS(const problem *prob, int nr_class, double *C, double eps, int max_iter)
{
	this->w_size = prob->n;
	this->l = prob->l;
	this->nr_class = nr_class;
	this->eps = eps;
	this->max_iter = max_iter;
	this->prob = prob;
	this->C = C;
	this->B = new double[nr_class];
	this->G = new double[nr_class];
}

Solver_MCSVM_CS::~Solver_MCSVM_CS()
{
	delete[] B;
	delete[] G;
}

int compare_double(const void *a, const void *b)
{
	if(*(double *)a > *(double *)b)
		return -1;
	if(*(double *)a < *(double *)b)
		return 1;
	return 0;
}

void Solver_MCSVM_CS::solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new)
{
	int r;
	double *D;

	clone(D, B, active_i);
	if(yi < active_i)
		D[yi] += A_i*C_yi;
	qsort(D, active_i, sizeof(double), compare_double);

	double beta = D[0] - A_i*C_yi;
	for(r=1;r<active_i && beta<r*D[r];r++)
		beta += D[r];

	beta /= r;
	for(r=0;r<active_i;r++)
	{
		if(r == yi)
			alpha_new[r] = min(C_yi, (beta-B[r])/A_i);
		else
			alpha_new[r] = min((double)0, (beta - B[r])/A_i);
	}
	delete[] D;
}

bool Solver_MCSVM_CS::be_shrunk(int m, int yi, double alpha_i, double minG)
{
	double bound = 0;
	if(m == yi)
		bound = C[yi];
	if(alpha_i == bound && G[m] < minG)
		return true;
	return false;
}

void Solver_MCSVM_CS::Solve(double *w)
{
	int i, m, s;

#ifdef _DENSE_REP
	int j;
#endif
	int iter = 0;
	double *alpha =  new double[l*nr_class];
	double *alpha_new = new double[nr_class];
	int *index = new int[l];
	double *QD = new double[l];
	int *d_ind = new int[nr_class];
	double *d_val = new double[nr_class];
	int *alpha_index = new int[nr_class*l];
	int *y_index = new int[l];
	int active_size = l;
	int *active_size_i = new int[l];
	double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	bool start_from_all = true;
	// initial
	for(i=0;i<l*nr_class;i++)
		alpha[i] = 0;
	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0; 
	for(i=0;i<l;i++)
	{
		for(m=0;m<nr_class;m++)
			alpha_index[i*nr_class+m] = m;
#ifdef _DENSE_REP
		double *xi = prob->x[i];

		QD[i] = 0;
		for(j = 0; j < w_size; j++)
		{
                        QD[i] += xi[j]*xi[j];
		}  
#else
		feature_node *xi = prob->x[i];
		QD[i] = 0;
		while(xi->index != -1)
		{
			QD[i] += (xi->value)*(xi->value);
			xi++;
		}
#endif
		active_size_i[i] = nr_class;
		y_index[i] = prob->y[i];
		index[i] = i;
	}

	while(iter < max_iter) 
	{
		double stopping = -INF;
		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		for(s=0;s<active_size;s++)
		{
			i = index[s];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*nr_class];
			int *alpha_index_i = &alpha_index[i*nr_class];

			if(Ai > 0)
			{
				for(m=0;m<active_size_i[i];m++)
					G[m] = 1;
				if(y_index[i] < active_size_i[i])
					G[y_index[i]] = 0;

#ifdef _DENSE_REP
				double *xi = prob->x[i];
				for(j=0; j < w_size ;j++)
				{
				        double *w_i = &w[j*nr_class];
					for(m=0;m<active_size_i[i];m++)
						G[m] += w_i[alpha_index_i[m]]*(xi[j]);
				}

#else
				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<active_size_i[i];m++)
						G[m] += w_i[alpha_index_i[m]]*(xi->value);
					xi++;
				}
#endif
				double minG = INF;
				double maxG = -INF;
				for(m=0;m<active_size_i[i];m++)
				{
					if(alpha_i[alpha_index_i[m]] < 0 && G[m] < minG)
						minG = G[m];
					if(G[m] > maxG)
						maxG = G[m];
				}
				if(y_index[i] < active_size_i[i])
					if(alpha_i[prob->y[i]] < C[prob->y[i]] && G[y_index[i]] < minG)
						minG = G[y_index[i]];

				for(m=0;m<active_size_i[i];m++)
				{
					if(be_shrunk(m, y_index[i], alpha_i[alpha_index_i[m]], minG))
					{
						active_size_i[i]--;
						while(active_size_i[i]>m)
						{
							if(!be_shrunk(active_size_i[i], y_index[i], 
											alpha_i[alpha_index_i[active_size_i[i]]], minG))
							{
								swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
								swap(G[m], G[active_size_i[i]]);
								if(y_index[i] == active_size_i[i])
									y_index[i] = m;
								else if(y_index[i] == m) 
									y_index[i] = active_size_i[i];
								break;
							}
							active_size_i[i]--;
						}
					}
				}

				if(active_size_i[i] <= 1)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;	
					continue;
				}

				if(maxG-minG <= 1e-12)
					continue;
				else
					stopping = max(maxG - minG, stopping);

				for(m=0;m<active_size_i[i];m++)
					B[m] = G[m] - Ai*alpha_i[alpha_index_i[m]] ;

				solve_sub_problem(Ai, y_index[i], C[prob->y[i]], active_size_i[i], alpha_new);
				int nz_d = 0;
				for(m=0;m<active_size_i[i];m++)
				{
					double d = alpha_new[m] - alpha_i[alpha_index_i[m]];
					alpha_i[alpha_index_i[m]] = alpha_new[m];
					if(fabs(d) >= 1e-12)
					{
						d_ind[nz_d] = alpha_index_i[m];
						d_val[nz_d] = d;
						nz_d++;
					}
				}

#ifdef _DENSE_REP
				xi = prob->x[i];
				for(j = 0; j < w_size; j++)
				{
					double *w_i = &w[(j)*nr_class];
					for(m=0;m<nz_d;m++)
						w_i[d_ind[m]] += d_val[m]*xi[j];
				}
#else
				xi = prob->x[i];
				while(xi->index != -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<nz_d;m++)
						w_i[d_ind[m]] += d_val[m]*xi->value;
					xi++;
				}
#endif
			}
		}

		iter++;
		if(iter % 10 == 0)
		{
			info(".");
		}

		if(stopping < eps_shrink)
		{
			if(stopping < eps && start_from_all == true)
				break;
			else
			{
				active_size = l;
				for(i=0;i<l;i++)
					active_size_i[i] = nr_class;
				info("*");
				eps_shrink = max(eps_shrink/2, eps);
				start_from_all = true;
			}
		}
		else
			start_from_all = false;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("Warning: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0;i<w_size*nr_class;i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0;i<l*nr_class;i++)
	{
		v += alpha[i];
		if(fabs(alpha[i]) > 0)
			nSV++;
	}
	for(i=0;i<l;i++)
		v -= alpha[i*nr_class+prob->y[i]];
	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] alpha_new;
	delete [] index;
	delete [] QD;
	delete [] d_ind;
	delete [] d_val;
	delete [] alpha_index;
	delete [] y_index;
	delete [] active_size_i;
}

// A coordinate descent algorithm for 
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= alpha_i <= upper_bound_i,
// 
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix 
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1
// 		D_ii = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		D_ii = 1/(2*Cp)	if y_i = 1
// 		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w

static void solve_l2r_l1l2_svc(
	const problem *prob, double *w, double eps, 
	double Cp, double Cn, int solver_type)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;

#ifdef _DENSE_REP
	int j;
#endif
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[l];
	double *alpha = new double[l];
	schar *y = new schar[l];
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag_p = 0.5/Cp, diag_n = 0.5/Cn;
	double upper_bound_p = INF, upper_bound_n = INF;
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		diag_p = 0; diag_n = 0;
		upper_bound_p = Cp; upper_bound_n = Cn;
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		alpha[i] = 0;
		if(prob->y[i] > 0)
		{
			y[i] = +1; 
			QD[i] = diag_p;
		}
		else
		{
			y[i] = -1;
			QD[i] = diag_n;
		}

#ifdef _DENSE_REP
		double *xi = prob->x[i];
		for(j = 0; j < w_size ; j++)
		{
		        QD[i] += xi[j]*xi[j];
		}
#else
		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			QD[i] += (xi->value)*(xi->value);
			xi++;
		}
#endif
		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;
		
		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0;s<active_size;s++)
		{
			i = index[s];
			G = 0;
			schar yi = y[i];

#ifdef _DENSE_REP
			double *xi = prob->x[i]; 
        		for(j = 0; j < w_size ; j++)
			{
				G += w[j]*xi[j];
			}
#else
			feature_node *xi = prob->x[i];
			while(xi->index!= -1)
			{
				G += w[xi->index-1]*(xi->value);
				xi++;
			}
#endif
			G = G*yi-1;

			if(yi == 1)
			{
				C = upper_bound_p; 
				G += alpha[i]*diag_p; 
			}
			else 
			{
				C = upper_bound_n;
				G += alpha[i]*diag_n; 
			}

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;
#ifdef _DENSE_REP
				xi = prob->x[i];
				for(j = 0; j < w_size; j++)
				{
					w[j] += d*xi[j];
				}
#else
				xi = prob->x[i];
				while (xi->index != -1)
				{
					w[xi->index-1] += d*xi->value;
					xi++;
				}
#endif	
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		if (y[i] == 1)
			v += alpha[i]*(alpha[i]*diag_p - 2); 
		else
			v += alpha[i]*(alpha[i]*diag_n - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

// A coordinate descent algorithm for 
// L1-regularized L2-loss support vector classification
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w

static void solve_l1r_l2_svc(
	problem *prob_col, double *w, double eps, 
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;

#ifdef _DENSE_REP
	int k;
#endif
	int max_iter = 1000;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double sigma = 0.01;
	double d, G_loss, G, H;
	double Gmax_old = INF;
	double Gmax_new;
	double Gmax_init;
	double d_old, d_diff;
	double loss_old, loss_new;
	double appxcond, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *b = new double[l]; // b = 1-ywTx
	double *xj_sq = new double[w_size];

#ifdef _DENSE_REP
	double *x;
#else
	feature_node *x;
#endif

	// To support weights for instances,
	// replace C[y[i]] with C[i].
	double C[2] = {Cn,Cp};

	for(j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = 0;
	}
	for(j=0; j<w_size; j++)
	{
		w[j] = 0;
		index[j] = j;
		xj_sq[j] = 0;

#ifdef _DENSE_REP
		x = prob_col->x[j]; 
		for(k = 0; k < l; k++)
		{
			double val = x[k];
			x[k] *= prob_col->y[k]; // x->value stores yi*xij
			xj_sq[j] += C[y[k]]*val*val;
		}
#else
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index;
			double val = x->value;
			x->value *= prob_col->y[ind]; // x->value stores yi*xij
			xj_sq[j] += C[y[ind]]*val*val;
			x++;
		}
#endif
	}

	while(iter < max_iter)
	{
		Gmax_new  = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			G_loss = 0;
			H = 0;

#ifdef _DENSE_REP
			x = prob_col->x[j];
			for(k = 0; k < l ; k++)
			{
				if(b[k] > 0)
				{
					double val = x[k];
					double tmp = C[y[k]]*val;
					G_loss -= tmp*b[k];
					H += tmp*val;
				}
			}
#else
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index;
				if(b[ind] > 0)
				{
					double val = x->value;
					double tmp = C[y[ind]]*val;
					G_loss -= tmp*b[ind];
					H += tmp*val;
				}
				x++;
			}
#endif
			G_loss *= 2;

			G = G_loss;
			H *= 2;
			H = max(H, 1e-12);

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);

			// obtain Newton direction d
			if(Gp <= H*w[j])
				d = -Gp/H;
			else if(Gn >= H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			d_old = 0;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				appxcond = xj_sq[j]*d*d + G_loss*d + cond;
				if(appxcond <= 0)
				{

#ifdef _DENSE_REP
					x = prob_col->x[j];
					for(k = 0; k < l; k++)
					{
         	                                b[k] += d_diff*x[k];
					}
#else
					x = prob_col->x[j];
					while(x->index != -1)
					{
						b[x->index] += d_diff*x->value;
						x++;
					}
#endif
					break;
				}

				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;
#ifdef _DENSE_REP
					x = prob_col->x[j];
					for(k = 0; k < l; k++)
					{
						if(b[k] > 0)
							loss_old += C[y[k]]*b[k]*b[k];
						double b_new = b[k] + d_diff*x[k];
						b[k] = b_new;
						if(b_new > 0)
							loss_new += C[y[k]]*b_new*b_new;
					}
				}
				else
				{
					loss_new = 0;
					x = prob_col->x[j];
					for(k = 0; k < l; k++)
					{
						double b_new = b[k] + d_diff*x[k];
						b[k] = b_new;
						if(b_new > 0)
							loss_new += C[y[k]]*b_new*b_new;
					}
#else
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index;
						if(b[ind] > 0)
							loss_old += C[y[ind]]*b[ind]*b[ind];
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[y[ind]]*b_new*b_new;
						x++;
					}
				}
				else
				{
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index;
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[y[ind]]*b_new*b_new;
						x++;
					}
#endif
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
					break;
				else
				{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info("#");
				for(int i=0; i<l; i++)
					b[i] = 1;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
#ifdef _DENSE_REP
					x = prob_col->x[i];
					for(int pp = 0; pp < l; pp++)
					{
						b[pp] -= w[i]*x[pp];
					}
#else
					x = prob_col->x[i];
					while(x->index != -1)
					{
						b[x->index] -= w[i]*x->value;
						x++;
					}
#endif
				}
			}
		}

		if(iter == 0)
			Gmax_init = Gmax_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gmax_new <= eps*Gmax_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
	{
#if _DENSE_REP
		x = prob_col->x[j];
		for(int k = 0; k < l; k++)
		{
			x[k] *= prob_col->y[k]; // restore x->value
		}
#else
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= prob_col->y[x->index]; // restore x->value
			x++;
		}
#endif
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(j=0; j<l; j++)
		if(b[j] > 0)
			v += C[y[j]]*b[j]*b[j];

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
}

// A coordinate descent algorithm for 
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w

static void solve_l1r_lr(
	const problem *prob_col, double *w, double eps, 
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;

#ifdef _DENSE_REP
	int k;
#endif
	int max_iter = 1000;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double x_min = 0;
	double sigma = 0.01;
	double d, G, H;
	double Gmax_old = INF;
	double Gmax_new;
	double Gmax_init;
	double sum1, appxcond1;
	double sum2, appxcond2;
	double cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *exp_wTx = new double[l];
	double *exp_wTx_new = new double[l];
	double *xj_max = new double[w_size];
	double *C_sum = new double[w_size];
	double *xjneg_sum = new double[w_size];
	double *xjpos_sum = new double[w_size];

#ifdef _DENSE_REP
	double *x;
#else
	feature_node *x;
#endif
	// To support weights for instances,
	// replace C[y[i]] with C[i].
	double C[2] = {Cn,Cp};

	for(j=0; j<l; j++)
	{
		exp_wTx[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = 0;
	}
	for(j=0; j<w_size; j++)
	{
		w[j] = 0;
		index[j] = j;
		xj_max[j] = 0;
		C_sum[j] = 0;
		xjneg_sum[j] = 0;
		xjpos_sum[j] = 0;
		x = prob_col->x[j];

#ifdef _DENSE_REP
		for(k = 0 ; k < l; k++)
		{
			double val = x[k];
			x_min = min(x_min, val);
			xj_max[j] = max(xj_max[j], val);
			C_sum[j] += C[y[k]];
			if(y[k] == 0)
				xjneg_sum[j] += C[y[k]]*val;
			else
				xjpos_sum[j] += C[y[k]]*val;
		}
#else
		while(x->index != -1)
		{
			int ind = x->index;
			double val = x->value;
			x_min = min(x_min, val);
			xj_max[j] = max(xj_max[j], val);
			C_sum[j] += C[y[ind]];
			if(y[ind] == 0)
				xjneg_sum[j] += C[y[ind]]*val;
			else
				xjpos_sum[j] += C[y[ind]]*val;
			x++;
		}
#endif
	}

	while(iter < max_iter)
	{
		Gmax_new = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			sum1 = 0;
			sum2 = 0;
			H = 0;

			x = prob_col->x[j];
#ifdef _DENSE_REP
			for(k = 0; k < l; k++)
			{
				double exp_wTxind = exp_wTx[k];
				double tmp1 = x[k]/(1+exp_wTxind);
				double tmp2 = C[y[k]]*tmp1;
				double tmp3 = tmp2*exp_wTxind;
				sum2 += tmp2;
				sum1 += tmp3;
				H += tmp1*tmp3;
			}
#else
			while(x->index != -1)
			{
				int ind = x->index;
				double exp_wTxind = exp_wTx[ind];
				double tmp1 = x->value/(1+exp_wTxind);
				double tmp2 = C[y[ind]]*tmp1;
				double tmp3 = tmp2*exp_wTxind;
				sum2 += tmp2;
				sum1 += tmp3;
				H += tmp1*tmp3;
				x++;
			}
#endif
			G = -sum2 + xjneg_sum[j];

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);

			// obtain Newton direction d
			if(Gp <= H*w[j])
				d = -Gp/H;
			else if(Gn >= H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			d = min(max(d,-10.0),10.0);

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				if(x_min >= 0)
				{
					double tmp = exp(d*xj_max[j]);
					appxcond1 = log(1+sum1*(tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond - d*xjpos_sum[j];
					appxcond2 = log(1+sum2*(1/tmp-1)/xj_max[j]/C_sum[j])*C_sum[j] + cond + d*xjneg_sum[j];
					if(min(appxcond1,appxcond2) <= 0)
					{
#ifdef _DENSE_REP
						x = prob_col->x[j];
						for(k = 0; k < l; k++)
						{
							exp_wTx[k] *= exp(d*x[k]);
						}
#else
						x = prob_col->x[j];
						while(x->index != -1)
						{
							exp_wTx[x->index] *= exp(d*x->value);
							x++;
						}
#endif
						break;
					}
				}

				cond += d*xjneg_sum[j];

				int i = 0;
				x = prob_col->x[j];
#ifdef _DENSE_REP
				for(k = 0; k < l; k++)
				{
					double exp_dx = exp(d*x[k]);
					exp_wTx_new[i] = exp_wTx[k]*exp_dx;
					cond += C[y[k]]*log((1+exp_wTx_new[i])/(exp_dx+exp_wTx_new[i]));
					i++;
				}
#else
				while(x->index != -1)
				{
					int ind = x->index;
					double exp_dx = exp(d*x->value);
					exp_wTx_new[i] = exp_wTx[ind]*exp_dx;
					cond += C[y[ind]]*log((1+exp_wTx_new[i])/(exp_dx+exp_wTx_new[i]));
					x++; i++;
				}
#endif
				if(cond <= 0)
				{
					int i = 0;
					x = prob_col->x[j];
#ifdef _DENSE_REP
					for(k = 0; k < l; k++)
					{
						exp_wTx[k] = exp_wTx_new[i];
						i++;
					}
#else
					while(x->index != -1)
					{
						int ind = x->index;
						exp_wTx[ind] = exp_wTx_new[i];
						x++; i++;
					}
#endif
					break;
				}
				else
				{
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute exp_wTx[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info("#");
				for(int i=0; i<l; i++)
					exp_wTx[i] = 0;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
					x = prob_col->x[i];
#ifdef _DENSE_REP
					for(k = 0; k < l; k++)
					{
						exp_wTx[k] += w[i]*x[k];
					}
#else
					while(x->index != -1)
					{
						exp_wTx[x->index] += w[i]*x->value;
						x++;
					}
#endif
				}

				for(int i=0; i<l; i++)
					exp_wTx[i] = exp(exp_wTx[i]);
			}
		}

		if(iter == 0)
			Gmax_init = Gmax_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gmax_new <= eps*Gmax_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	
	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	for(j=0; j<l; j++)
		if(y[j] == 1)
			v += C[y[j]]*log(1+1/exp_wTx[j]);
		else
			v += C[y[j]]*log(1+exp_wTx[j]);

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] exp_wTx;
	delete [] exp_wTx_new;
	delete [] xj_max;
	delete [] C_sum;
	delete [] xjneg_sum;
	delete [] xjpos_sum;
}

// transpose matrix X from row format to column format
#ifdef _DENSE_REP
static void transpose(const problem *prob, double **x_space_ret, problem *prob_col)
#else
static void transpose(const problem *prob, feature_node **x_space_ret, problem *prob_col)
#endif
{
	int i;
	int l = prob->l;
	int n = prob->n;
	int nnz = 0;

	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = new int[l];
	prob_col->bias = prob->bias;

#ifdef _DENSE_REP
	double *x_space;
	prob_col->x = new double*[n];
#else
	int *col_ptr = new int[n+1];
	feature_node *x_space;
	prob_col->x = new feature_node*[n];
#endif

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

#ifdef _DENSE_REP
	nnz = l*n;
	x_space = new double[nnz];

	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[i*l];

	//simply transpose the data
	for(i=0; i<l; i++)
	{
	  	double *x = prob->x[i];
	        for(int j=0; j<n; j++)
		{
		  x_space[i + j*l] = x[j];
		}
	}
	*x_space_ret = x_space;
#else

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;

	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new feature_node[nnz+n];

	for(i=0; i < n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i;
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;
	*x_space_ret = x_space;

	delete [] col_ptr;
#endif
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn)
{
	double eps=param->eps;
	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob->l;i++)
		if(prob->y[i]==+1)
			pos++;
	neg = prob->l - pos;

	function *fun_obj=NULL;
	//print_prob(prob);
	switch(param->solver_type)
	{
		case L2R_LR:
		{
			fun_obj=new l2r_lr_fun(prob, Cp, Cn);
			TRON tron_obj(fun_obj, eps*min(pos,neg)/prob->l);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			break;
		}
		case L2R_L2LOSS_SVC:
		{
    		       
			fun_obj=new l2r_l2_svc_fun(prob, Cp, Cn);
			TRON tron_obj(fun_obj, eps*min(pos,neg)/prob->l);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			break;
		}
		case L2R_L2LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
			break;
		case L2R_L1LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			break;
		case L1R_L2LOSS_SVC:
		{
			problem prob_col;
#ifdef _DENSE_REP
			double *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			//print_prob_col(&prob_col);
#else
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
#endif

			solve_l1r_l2_svc(&prob_col, w, eps*min(pos,neg)/prob->l, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L1R_LR:
		{
			problem prob_col;
#ifdef _DENSE_REP
			double *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			//print_prob_col(&prob_col);
#else
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
#endif
			solve_l1r_lr(&prob_col, w, eps*min(pos,neg)/prob->l, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		default:
			fprintf(stderr, "Error: unknown solver_type\n");
			break;
	}
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	int nr_class;
	int *label = NULL;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int,l);

	// group training data of the same class
	group_classes(prob,&nr_class,&label,&start,&count,perm);

	model_->nr_class=nr_class;
	model_->label = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++)
		model_->label[i] = label[i];

	// calculate weighted C
	double *weighted_C = Malloc(double, nr_class);
	for(i=0;i<nr_class;i++)
		weighted_C[i] = param->C;
	for(i=0;i<param->nr_weight;i++)
	{
		for(j=0;j<nr_class;j++)
			if(param->weight_label[i] == label[j])
				break;
		if(j == nr_class)
			fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
		else
			weighted_C[j] *= param->weight[i];
	}

	// constructing the subproblem
#ifdef _DENSE_REP
	double **x = Malloc(double *,l);
#else
	feature_node **x = Malloc(feature_node *,l);
#endif

	for(i=0;i<l;i++)
		x[i] = prob->x[perm[i]];

	int k;
	problem sub_prob;
	sub_prob.l = l;
	sub_prob.n = n;
	sub_prob.bias = prob->bias;

#ifdef _DENSE_REP
	sub_prob.x = Malloc(double *,sub_prob.l);
#else
	sub_prob.x = Malloc(feature_node *,sub_prob.l);
#endif	
	sub_prob.y = Malloc(int,sub_prob.l);



	for(k=0; k<sub_prob.l; k++)
		sub_prob.x[k] = x[k];

	// multi-class svm by Crammer and Singer
	if(param->solver_type == MCSVM_CS)
	{
		model_->w=Malloc(double, n*nr_class);
		for(i=0;i<nr_class;i++)
			for(j=start[i];j<start[i]+count[i];j++)
				sub_prob.y[j] = i;
		Solver_MCSVM_CS Solver(&sub_prob, nr_class, weighted_C, param->eps);
		Solver.Solve(model_->w);
	}
	else
	{
		if(nr_class == 2)
		{
			model_->w=Malloc(double, w_size);

			int e0 = start[0]+count[0];
			k=0;
			for(; k<e0; k++)
				sub_prob.y[k] = +1;
			for(; k<sub_prob.l; k++)
				sub_prob.y[k] = -1;

			train_one(&sub_prob, param, &model_->w[0], weighted_C[0], weighted_C[1]);
		}
		else
		{
			model_->w=Malloc(double, w_size*nr_class);
			double *w=Malloc(double, w_size);
			for(i=0;i<nr_class;i++)
			{
				int si = start[i];
				int ei = si+count[i];

				k=0;
				for(; k<si; k++)
					sub_prob.y[k] = -1;
				for(; k<ei; k++)
					sub_prob.y[k] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;

				train_one(&sub_prob, param, w, weighted_C[i], param->C);

				for(int j=0;j<w_size;j++)
					model_->w[j*nr_class+i] = w[j];
			}
			free(w);
		}

	}

	free(x);
	free(label);
	free(start);
	free(count);
	free(perm);
	free(sub_prob.x);
	free(sub_prob.y);
	free(weighted_C);
	return model_;
}

void destroy_model(struct model *model_)
{
	if(model_->w != NULL)
		free(model_->w);
	if(model_->label != NULL)
		free(model_->label);
	free(model_);
}

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC","L2R_L1LOSS_SVC_DUAL","MCSVM_CS", "L1R_L2LOSS_SVC","L1R_LR", NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	int nr_w;
	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);
	fprintf(fp, "label");
	for(i=0; i<model_->nr_class; i++)
		fprintf(fp, " %d", model_->label[i]);
	fprintf(fp, "\n");

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				free(model_->label);
				free(model_);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model_);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
		fscanf(fp, "\n");
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}
#ifdef _DENSE_REP
int predict_values(const struct model *model_, const double *x, double *dec_values)
#else
int predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
#endif
{

#ifdef _DENSE_REP
        int k;
#else
	int idx;
#endif
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;

#ifdef _DENSE_REP
	const double *lx = x;
	for(k = 0; k < n; k++)
	{
	        for(i=0;i<nr_w;i++)
        		  dec_values[i] += w[k*nr_w+i]*lx[k];
	}
#else
	const feature_node *lx=x;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}
#endif
	if(nr_class==2)
		return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

#ifdef _DENSE_REP
int predict(const model *model_, const double *x)
#else
int predict(const model *model_, const feature_node *x)
#endif

{
	double *dec_values = Malloc(double, model_->nr_class);
	int label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}
#ifdef _DENSE_REP
int predict_probability(const struct model *model_, const double *x,double* prob_estimates)
#else
int predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
#endif
{
	if(model_->param.solver_type==L2R_LR)
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		int label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;		
	}
	else
		return 0;
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->solver_type != L2R_LR
		&& param->solver_type != L2R_L2LOSS_SVC_DUAL
		&& param->solver_type != L2R_L2LOSS_SVC
		&& param->solver_type != L2R_L1LOSS_SVC_DUAL
		&& param->solver_type != MCSVM_CS
		&& param->solver_type != L1R_L2LOSS_SVC
		&& param->solver_type != L1R_LR)
		return "unknown solver type";

	return NULL;
}

void cross_validation(const problem *prob, const parameter *param, int nr_fold, int *target)
{
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int *perm = Malloc(int,l);

	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct problem subprob;

		subprob.bias = prob->bias;
		subprob.n = prob->n;
		subprob.l = l-(end-begin);

#ifdef _DENSE_REP
		subprob.x = Malloc(double *,subprob.l);
#else
		subprob.x = Malloc(struct feature_node*,subprob.l);
#endif
		subprob.y = Malloc(int,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct model *submodel = train(&subprob,param);
		for(j=begin;j<end;j++)
			target[perm[j]] = predict(submodel,prob->x[perm[j]]);
		destroy_model(submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

