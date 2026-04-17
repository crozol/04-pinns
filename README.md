# Physics-Informed Neural Networks (PINNs)

Resolver ecuaciones en derivadas parciales (PDEs) usando redes neuronales cuyo loss incluye el residual de la PDE (Raissi, Perdikaris & Karniadakis, 2019).

## Motivación

Un PINN es una red neuronal `u_θ(x, t)` cuyo loss combina:

1. **Condiciones iniciales / de frontera** (datos observados u impuestos).
2. **Residual de la PDE** evaluado en puntos del dominio, calculado con `autograd`.

El modelo aprende a satisfacer la PDE en sentido clásico sin necesidad de mallado como un método numérico tradicional. Este proyecto implementa:

- **Ecuación del calor 1D**: como caso mínimo para validar contra la solución analítica.
- **Ecuación de Burgers 1D**: caso no lineal clásico, con discontinuidad emergente.

## Stack

- Python 3.11+
- PyTorch 2.x (autograd para derivadas parciales)
- NumPy, SciPy (soluciones de referencia)
- Matplotlib

## Estructura

```
04-pinns/
├── README.md
├── requirements.txt
├── src/
│   ├── pinn.py           # Red neuronal u_theta(x, t)
│   ├── physics.py        # Residuales PDE (heat, Burgers)
│   └── train.py          # Training loop con loss compuesto
└── notebooks/
```

## Roadmap

- [ ] Implementar MLP con activación `tanh` (diferenciable dos veces).
- [ ] Función `residual_heat(u, x, t)` usando `torch.autograd.grad`.
- [ ] Función `residual_burgers(u, x, t, nu)`.
- [ ] Training loop que muestrea puntos de collocation y enforce boundary/initial.
- [ ] Validación contra solución analítica (heat) y solución numérica (Burgers).
- [ ] Gráficas espacio-tiempo + blog post.

## Resultado esperado

Error L2 relativo < 1% contra la solución analítica de la ecuación del calor, y < 5% contra la solución numérica de referencia en Burgers.
