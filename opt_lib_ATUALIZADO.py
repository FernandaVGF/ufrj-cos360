import numpy as np

# Função f (x1, x2)

def f(ponto):
	"""Retorna o valor da nossa funcao avaliada no ponto passado como argumento"""

	x = ponto[0]
	y = ponto[1]
	
	h = y*(1-x)*(1+y)*x
	
	assert 0 < h <= 1, "O ponto nao pertence ao dominio de f."
	
	g = -np.log(h)

	return np.sqrt(g)

def gradf(ponto):
	"""Retorna o gradiente da nossa função avaliada no ponto passado como argumento"""

	x = ponto[0]
	y = ponto[1]

	h = y*(1-x)*(1+y)*x
	assert 0 < h < 1, "O ponto nao pertence ao dominio de f."
	assert x != 0, "O ponto nao pertence ao dominio de f."
	assert x != 1, "O ponto nao pertence ao dominio de f."
	assert y != 0, "O ponto nao pertence ao dominio de f."
	assert y != -1, "O ponto nao pertence ao dominio de f."

	num_1 = 1-2*x
	den_1 = 2*(x-1)*x*np.sqrt(-np.log(h))

	num_2 = -(2*y+1)
	den_2 = 2*(y+1)*y*np.sqrt(-np.log(h))

	grad = np.array([num_1/den_1, num_2/den_2])

	return grad

def hessf(ponto):
	"""Retorna a hessiana da nossa função avaliada no ponto passado como argumento"""

	x = ponto[0]
	y = ponto[1]

	h = y*(1-x)*(1+y)*x
	assert 0 < h < 1, "O ponto nao pertence ao dominio de f."
	assert x != 1, "O ponto nao pertence ao dominio de f."
	assert x != 0, "O ponto nao pertence ao dominio de f."
	assert y != 0, "O ponto nao pertence ao dominio de f."
	assert y != -1, "O ponto nao pertence ao dominio de f."

	a_num = -((4*pow(x,2) - 4*x +2)*np.log(h) + pow(1 - 2*x, 2))
	a_den = 4*pow(x - 1, 2)*pow(x, 2)*pow(-np.log(h), 3/2)
	a = a_num/a_den

	b_num = (2*x - 1)*(2*y + 1)
	b_den = 4*h*pow(-np.log(h), 3/2)
	b = b_num/b_den

	c_num = (2*x - 1)*(2*y + 1)
	c_den = 4*h*pow(-np.log(h), 3/2)
	c = c_num/c_den

	d_num = -(pow((1 + 2*y), 2) + (2 + 4*y + 4*pow(y, 2))*np.log(h))
	d_den = 4*pow(y, 2)*pow((1 + y), 2)*pow(-np.log(h), 3/2)
	d = d_num/d_den 

	hess = np.array([[a,b],[c,d]])

	return hess


# Critérios de convergência usados

def delta_x(parametros):
	
	tolerancia = parametros['tol_conv']
	x_atual = parametros['x_atual']
	x_anterior = parametros['x_anterior']

	deltax = x_atual - x_anterior

	norma = np.linalg.norm(deltax)
	
	return norma < tolerancia

def delta_f(parametros):
	
	tolerancia = parametros['tol_conv']
	f_atual = parametros['f_atual']
	f_anterior = parametros['f_anterior']

	deltaf = f_atual - f_anterior

	norma = np.linalg.norm(deltaf)

	return norma < tolerancia


# Critérios de convergência extras
    
def k_max(parametros):
	
	tolerancia = parametros['tol_conv']
	k_atual = parametros['k_atual']
	return k_atual > tolerancia

def estacionario(parametros):
	
	tolerancia = parametros['tol_conv']
	gradiente = parametros['gradiente']

	return np.linalg.norm(gradiente) < tolerancia


# Métodos de busca unidirecional (encontrar t)

def secao_aurea(parametros):
    i = 1
    x = parametros['x']
    tolerancia = parametros['tolerancia']
    funcao = parametros['funcao']
    d = parametros['d']
    p = parametros['p']
    theta1 = parametros['theta1']
    theta2 = parametros['theta2']
    
    
    #print('x1 = %.2f\nx2 = %.2f\nd1 = %.2f\nd2 = %.2f'%(x[0],x[1],d[0],d[1]))
    
    assert tolerancia > 0, "Tolerancia deve ser maior que zero."
    assert p > 0 , "Parametro p deve ser maior que zero"
    
    # Obtendo intervalo
    a = 0
    s = p
    b = 2 * p
    
    # Testando se a funcao eh valida se t = b
    try:
        funcao(x + s * d)
    except AssertionError:
        while True:
            s = s - (s-a)*0.1
            try:
                funcao(x + s * d)
                return s
            except AssertionError:
                pass
            i += 1
				
    try:
        while funcao(x + b * d) < funcao(x + s * d):
            a = s
            s = b
            b = 2 * b
            i += 1
    except AssertionError:
        while True:
            b = b - (b-s)*0.1
            try:
                funcao(x + b * d)
                return b
            except AssertionError:
                pass
            i += 1

    # Obtencao do t*
    u = a + theta1 * (b-a)
    v = a + theta2 * (b-a)
    
    while (b-a) > tolerancia:
        phi_u = funcao(x + u * d)
        phi_v = funcao(x + v * d)
    
        if phi_u < phi_v:
            b = v
            v = u
            u = a + theta1 * (b-a)
        else:
            a = u
            u = v
            v = a + theta2 * (b-a)
    
    return (u+v)/2

def armijo(parametros):
	
    x = parametros['x']
    funcao = parametros['funcao']
    gradiente = parametros['gradiente']
    d = parametros['d']
    gama = parametros['gama']
    eta = parametros['eta']
    
    assert 0 < gama < 1, "Parametro gama deve estar no intervalo (0,1)."
    assert 0 < eta < 1, "Parametro eta deve estar no intervalo (0,1)."
    
    t = 1

    while True:
        try:
            f = funcao(x + t * d)
            f_aprox = funcao(x) + (eta * t * np.dot(gradiente(x), d))

            while f > f_aprox:
                t = gama * t
                f = funcao(x + t * d)
                f_aprox = funcao(x) + (eta * t * np.dot(gradiente(x), d))

            break

        except AssertionError:
            t = gama * t
	
    return t


# Metodos de obtencao de minimos

def metodo_gradiente(x0, funcao, gradiente, hessiana, param_convergencia, param_passo):
	
	param_convergencia = param_convergencia
	param_passo = param_passo
	convergencia = param_convergencia['func_convergencia']
	obter_passo = param_passo['func_passo']

	k = 0
	x_atual = x0
	x_anterior = np.array([np.inf for i in range(len(x0))])
	f_atual = funcao(x_atual)
	f_anterior = np.inf
	grad = gradiente(x_atual)

	while convergencia(param_convergencia) != True:
		d = -grad
		param_passo['d'] = d
		t = obter_passo(param_passo)
		x_anterior = x_atual
		x_atual = x_atual + t * d
		param_passo['x'] = x_atual
		f_anterior = f_atual
		f_atual = funcao(x_atual)
		grad = gradiente(x_atual)
		k = k + 1

		# Atualizando os parametros de convergencia e de busca do passo
		
		param_convergencia['x_atual'] = x_atual
		param_convergencia['x_anterior'] = x_anterior
		param_convergencia['f_atual'] = f_atual
		param_convergencia['f_anterior'] = f_anterior
		param_convergencia['k_atual'] = k
		param_convergencia['gradiente'] = grad

	return k, x_atual, f_atual

def metodo_Newton(x0, funcao, gradiente, hessiana, param_convergencia, param_passo):
	
    # Importante lembrar que a direcao dk pode nao estar definida ou nao ser de descida
    param_convergencia = param_convergencia
    param_passo = param_passo
    convergencia = param_convergencia['func_convergencia']
    obter_passo = param_passo['func_passo']
    
    k = 0
    x_atual = x0
    x_anterior = np.array([np.inf for i in range(len(x0))])
    f_atual = funcao(x_atual)
    f_anterior = np.inf
    grad = gradiente(x_atual)
    
    while convergencia(param_convergencia) != True:
        d = np.linalg.solve((hessiana(x_atual)),-grad)
        if not np.dot(grad, d) < 0: # subida
            d = -d
        param_passo['d'] = d
        t = obter_passo(param_passo)
        x_anterior = x_atual
        x_atual = x_atual + t * d
        param_passo['x'] = x_atual
        f_anterior = f_atual
        f_atual = funcao(x_atual)
        grad = gradiente(x_atual)
        k = k + 1
        
        # Atualizando os parametros de convergencia e de busca do passo
        		
        param_convergencia['x_atual'] = x_atual
        param_convergencia['x_anterior'] = x_anterior
        param_convergencia['f_atual'] = f_atual
        param_convergencia['f_anterior'] = f_anterior
        param_convergencia['k_atual'] = k
        param_convergencia['gradiente'] = grad
    
    return k, x_atual, f_atual

def BFGS(x_atual, x_anterior, gradiente, Hk):
	p = x_atual - x_anterior
	q = gradiente(x_atual) - gradiente(x_anterior)
	q_transp = np.transpose(q)
	p_transp = np.transpose(p)

	ptq = np.dot(p_transp, q)
	qtHq_div_ptq = np.dot(np.dot(q_transp, Hk),q)/ptq
	ppt_div_ptq = np.dot(p, p_transp)/ptq
	pqtH_div_pqt = np.dot(np.dot(p, q_transp),Hk)/ptq
	Hqpt_div_pqt = np.dot(np.dot(Hk, q),p_transp)/ptq

	H = Hk + ((1 + qtHq_div_ptq) * ppt_div_ptq) - (pqtH_div_pqt + Hqpt_div_pqt)

	return H


def metodo_quase_Newton(x0, funcao, gradiente, param_convergencia, param_passo):
	
    # Como Hk eh definida positiva, então dk eh bem definido e sempre de descida
    param_convergencia = param_convergencia
    param_passo = param_passo
    convergencia = param_convergencia['func_convergencia']
    obter_passo = param_passo['func_passo']
	
    k = 0
    x_atual = x0
    x_anterior = np.array([np.inf for i in range(len(x0))])
    f_atual = funcao(x_atual)
    f_anterior = np.inf
    grad = gradiente(x_atual)
    H = np.identity(len(x0))

    while convergencia(param_convergencia) != True:
        d = -np.dot(H,grad)
        if not np.dot(grad, d) < 0: # subida
            d = -d
        param_passo['d'] = d
        t = obter_passo(param_passo)
        x_anterior = x_atual
        x_atual = x_atual + t * d
        param_passo['x'] = x_atual
        f_anterior = f_atual
        f_atual = funcao(x_atual)
        grad = gradiente(x_atual)
        H = BFGS(x_atual, x_anterior, gradiente, H)
        k = k + 1 

        # Atualizando os parametros de convergencia e de busca do passo
		
        param_convergencia['x_atual'] = x_atual
        param_convergencia['x_anterior'] = x_anterior
        param_convergencia['f_atual'] = f_atual
        param_convergencia['f_anterior'] = f_anterior
        param_convergencia['k_atual'] = k
        param_convergencia['gradiente'] = grad

    return k, x_atual, f_atual


# Passagem de parâmetros para critérios de convergência

def parametros_convergencia(id_criterio, *args):

	criterios = {
		'k_max':['tol_conv','k_atual'],
		'delta_x':['tol_conv','x_atual', 'x_anterior'],
		'delta_f':['tol_conv','f_atual', 'f_anterior'],
		'estacionario':['tol_conv','gradiente']
	}

	param_convergencia = {'func_convergencia':eval(id_criterio)}

	for i, parametro in enumerate(criterios[id_criterio]):
		param_convergencia[parametro] = args[i]

	return param_convergencia


# Passagem de parâmetros para os métodos de busca unidirecional

def parametros_passo(id_metodo, *args):
	metodos = {
		'secao_aurea':['x','tolerancia','funcao','d','p','theta1','theta2'],
		'armijo':['x','funcao', 'gradiente', 'd', 'gama','eta']
	}
		
	param_passo = {'func_passo':eval(id_metodo)}

	for i, parametro in enumerate(metodos[id_metodo]):
		param_passo[parametro] = args[i]

	return param_passo


# Testes
    
theta1 = (3-np.sqrt(5))/2
theta2 = 1-theta1

## Caso

#x = np.array([0.5,0.5])

## Caso 1 (z = 0.1)

#x = np.array([0.95,1.04])

## Caso 2

#x = np.array([2.009,-0.948])

## Caso 3

x = np.array([0.836,-1.49])

## Caso 4

#x = np.array([-0.997,-0.947])

d = np.array([np.inf,np.inf])
tol_conv = 0.0001

#p = 0.5
p = 0.1 # para quase-Newton
tol_passo = 0.001
aurea_param = parametros_passo('secao_aurea',x,tol_passo,f,d,p,theta1,theta2)

gama = 0.9
#eta = 0.0001
eta = 0.9
armijo_param = parametros_passo('armijo',x,f,gradf,d,gama,eta)

f_atual = f(x)
f_anterior = np.inf
x_atual = x
x_anterior = np.array([np.inf,np.inf])
deltaf_param = parametros_convergencia('delta_f',tol_conv,f_atual,f_anterior)
deltax_param = parametros_convergencia('delta_x',tol_conv,x_atual,x_anterior)
kmax_param = parametros_convergencia('k_max',tol_conv,0)
    
## Gradiente (seção áurea)

#print(metodo_gradiente(x,f,gradf,hessf,deltaf_param,aurea_param))
#print(metodo_gradiente(x,f,gradf,hessf,deltax_param,aurea_param))

## Gradiente (Armijo)

#print(metodo_gradiente(x,f,gradf,hessf,deltaf_param,armijo_param))
#print(metodo_gradiente(x,f,gradf,hessf,deltax_param,armijo_param))

## Newton (seção áurea)

#print(metodo_Newton(x,f,gradf,hessf,deltaf_param,aurea_param))
#print(metodo_Newton(x,f,gradf,hessf,deltax_param,aurea_param))

## Newton (Armijo)

#print(metodo_Newton(x,f,gradf,hessf,deltax_param,armijo_param))
#print(metodo_Newton(x,f,gradf,hessf,deltaf_param,armijo_param))

# Quase-Newton (seção áurea):

#print(metodo_quase_Newton(x,f,gradf,deltax_param,aurea_param))
#print(metodo_quase_Newton(x,f,gradf,deltaf_param,aurea_param))

# Quase-Newton (Armijo):

#print(metodo_quase_Newton(x,f,gradf,deltax_param,armijo_param))
#print(metodo_quase_Newton(x,f,gradf,deltaf_param,armijo_param))