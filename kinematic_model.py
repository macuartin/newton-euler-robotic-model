from contextlib import redirect_stdout
import sympy as sp
import utils
from tqdm import tqdm


def homogeneus_matrix(DHi):
    '''
    Recibe una fila de la matriz DH (link)
    '''

    alpha = DHi[4]
    a = DHi[3]
    link_type = DHi[0]

    if(link_type == 'rotational'):
        d = DHi[2]
        theta = sp.symbols(DHi[1])
        p = [a*sp.cos(theta), a*sp.sin(theta), d]
    elif(link_type == 'prismatic'):
        d = sp.symbols(DHi[2])
        theta = DHi[1]
        p = [0, 0, d]

    A = [[sp.cos(theta), -sp.cos(alpha)*sp.sin(theta),
         sp.sin(alpha)*sp.sin(theta), p[0]],
         [sp.sin(theta),  sp.cos(alpha)*sp.cos(theta),
         -sp.sin(alpha)*sp.cos(theta), p[1]],
         [0, sp.sin(alpha), sp.cos(alpha), p[2]],
         [0, 0, 0, 1]]

    return sp.Matrix(A)


def forward_kinematic(DH):
    '''
    Recibe la matriz DH
    '''

    T = sp.eye(4)

    for link in DH:
        A = homogeneus_matrix(link).evalf(n=4, chop=True)
        T = T * A

    return T


def jacobian_column(T):
    '''
    Recibe la matriz T y devuelve una columna de la matriz jacobiana
    '''

    J = sp.Matrix([[0] * 1] * 6)

    J[0] = -T[0, 0]*T[1, 3]+T[1, 0]*T[0, 3]
    J[1] = -T[0, 1]*T[1, 3]+T[1, 1]*T[0, 3]
    J[2] = -T[0, 2]*T[1, 3]+T[1, 2]*T[0, 3]
    J[3] = T[2, 0]
    J[4] = T[2, 1]
    J[5] = T[2, 2]

    return J


if __name__ == '__main__':

    # [type, theta, d, a, alpha]
    DH = [['rotational', 'q1', 0.147, 0.033, 1.5707],
          ['rotational', 'q2', 0, 0.155, 0],
          ['rotational', 'q3', 0, 0.135, 0],
          ['rotational', 'q4', 0, 0, 1.5707],
          ['rotational', 'q5', 0.2175, 0, 0]]

    degrees_of_freedom = len(DH)
    pbar = tqdm(total=100, desc='Kinematic Model')
    with open('kinematic_model.txt', 'w') as f:

        with redirect_stdout(f):
            print('Denavit-Hartenberg Model')
            print('[type, theta, d, a, alpha]')
            sp.pprint(sp.Matrix(DH))
            print()

        while len(DH) != 0:
            T = forward_kinematic(DH)
            J = jacobian_column(T)

            with redirect_stdout(f):
                print('Matrix T{}'.format(len(DH)))
                sp.pprint(sp.simplify(T), wrap_line=False)
                print()
                print('Jacobian Column for T{}'.format(len(DH)))
                sp.pprint(sp.simplify(J), wrap_line=False)
                print()

            pbar.update(100/degrees_of_freedom)
            del DH[0]
    pbar.close()
