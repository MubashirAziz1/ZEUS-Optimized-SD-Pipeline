import torch

from dataclasses import dataclass
from typing import Optional, Union, Tuple

from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from diffusers import DPMSolverMultistepScheduler

def lagrange_skip(t_points, x_values, t_eval):
    P_of_t = torch.zeros_like(x_values[0])
    n = len(t_points)

    for i in range(n):
        term = x_values[i].clone()
        for j in range(n):
            if j != i:
                factor = (t_eval - t_points[j]) / (t_points[i] - t_points[j])
                term = term * factor
        P_of_t += term

    return P_of_t


@dataclass
class SchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor

class PatchedDPMSolverMultistepScheduler(DPMSolverMultistepScheduler):

  def step(
          self,
          model_output: torch.Tensor,
          timestep: Union[int, torch.Tensor],
          sample: torch.Tensor,
          generator=None,
          variance_noise: Optional[torch.Tensor] = None,
          return_dict: bool = True,
  ) -> Union[SchedulerOutput, Tuple]:
      if self.step_index is None:
          self._init_step_index(timestep)

      # Improve numerical stability for small number of steps
      lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
              self.config.euler_at_final
              or (self.config.lower_order_final and len(self.timesteps) < 15)
              or self.config.final_sigmas_type == "zero"
      )
      lower_order_second = (
              (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(
          self.timesteps) < 15
      )

      # == Cache epsilon before Data Reconstruction == #
      for i in range(1):
          self._cache_bus.prev_epsilon_guided[i] = self._cache_bus.prev_epsilon_guided[i + 1]
      self._cache_bus.prev_epsilon_guided[-1] = model_output.clone()

      # == Conduct the step skipping identified from last step == #
      if self._cache_bus.skip_this_step and self._cache_bus.pred_m_m_1 is not None:
          model_output = self._cache_bus.pred_m_m_1
          self._cache_bus.skip_this_step = False
      else:
          model_output = self.convert_model_output(model_output, sample=sample)

      if self._cache_bus._tome_info['args']['lagrange_term'] != 0:
          use_lagrange = True

          lagrange_term = self._cache_bus._tome_info['args']['lagrange_term']
          lagrange_step = self._cache_bus._tome_info['args']['lagrange_step']
          lagrange_int = self._cache_bus._tome_info['args']['lagrange_int']

          if self._step_index % lagrange_int == 1:
              for i in range(lagrange_term - 1):
                  self._cache_bus.lagrange_x0[i] = self._cache_bus.lagrange_x0[i + 1]
                  self._cache_bus.lagrange_step[i] = self._cache_bus.lagrange_step[i + 1]
              self._cache_bus.lagrange_x0[-1] = model_output
              self._cache_bus.lagrange_step[-1] = self._step_index

      else: use_lagrange = False

      sigma_t, sigma_s0, sigma_s1 = (
          self.sigmas[self.step_index + 1],
          self.sigmas[self.step_index],
          self.sigmas[self.step_index - 1],
      )

      N = self.timesteps.shape[0]
      delta = 1 / N  # step size correlates with number of inference step

      interp_mode = self._cache_bus._tome_info['args']['interp_mode']
      cache_mode = self._cache_bus._tome_info['args']['caching_mode']

      alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
      beta_n = self.betas[self.timesteps[self.step_index]]
      s_alpha_cumprod_n = self.alpha_t[self.timesteps[self.step_index]]
      epsilon_0, epsilon_1 = self._cache_bus.prev_epsilon_guided[-1], self._cache_bus.prev_epsilon_guided[-2]

      # cache-retrieving patch
      if cache_mode == "interp_all" and interp_mode == "psi" and self._cache_bus.cons_skip > 0:
          epsilon_0 = self._cache_bus.prev_interp
          self._cache_bus.prev_epsilon_guided[-1] = self._cache_bus.prev_interp.clone()

      # == Criteria == #
      if self.config.prediction_type == "epsilon":
          f = (- 0.5 * beta_n * N * sample) + (0.5 * beta_n * N / sigma_s0) * epsilon_0
      elif self.config.prediction_type == "v_prediction":
          f = (0.5 * beta_n * N / sigma_s0) * s_alpha_cumprod_n * epsilon_0
      else: raise RuntimeError

      if self._cache_bus.prev_f[0] is not None:
          max_interval = self._cache_bus._tome_info['args']['max_interval']
          acc_range = self._cache_bus._tome_info['args']['acc_range']
          denominator = self._cache_bus._tome_info['args']['denominator']
          modular = self._cache_bus._tome_info['args']['modular']

          lagrange_this_step = False

          if self._cache_bus.cons_skip >= max_interval:
              self._cache_bus.skip_this_step = False
              self._cache_bus.cons_skip = 0

          elif not use_lagrange and (self._step_index % denominator in modular and self._step_index in range(acc_range[0], acc_range[1])):
              # == Here we skip step with psi / x0 interpolation == #
              self._cache_bus.skip_this_step = True
              self._cache_bus.cons_skip += 1
              self._cache_bus.skipping_path.append(self._step_index)

          elif use_lagrange and (self._step_index % denominator in modular and self._step_index in range(acc_range[0], lagrange_step)):
              # == Here we skip step with psi / x0 interpolation == #
              self._cache_bus.skip_this_step = True
              self._cache_bus.cons_skip += 1
              self._cache_bus.skipping_path.append(self._step_index)

          elif use_lagrange and (self._step_index % lagrange_int != 0 and self._step_index in range(lagrange_step, acc_range[1])):
              # == Here we skip step using lagrange == #
              self._cache_bus.skip_this_step = True
              self._cache_bus.cons_skip += 1
              self._cache_bus.skipping_path.append(self._step_index)
              lagrange_this_step = True

          else:
              self._cache_bus.skip_this_step = False
              self._cache_bus.cons_skip = 0

      for i in range(self.config.solver_order - 1):
          self.model_outputs[i] = self.model_outputs[i + 1]
      self.model_outputs[-1] = model_output

      # Upcast to avoid precision issues when computing prev_sample
      sample = sample.to(torch.float32)
      if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"] and variance_noise is None:
          noise = randn_tensor(
              model_output.shape, generator=generator, device=model_output.device, dtype=torch.float32
          )
      elif self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
          noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
      else:
          noise = None

      if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
          prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
      elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
          prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
      else:
          prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample, noise=noise)

      if self.lower_order_nums < self.config.solver_order:
          self.lower_order_nums += 1

      # Cast sample back to expected dtype
      prev_sample = prev_sample.to(model_output.dtype)

      # upon completion increase step index by one
      self._step_index += 1

      # == approximate next step data reconstruction == #
      if self._cache_bus.skip_this_step:
          if not lagrange_this_step:
              if interp_mode == "psi":
                  # interpolate on model output (psi)
                  epsilon_m1 = epsilon_0 + (epsilon_0 - epsilon_1)

                  if cache_mode == "interp_all" or (
                      cache_mode == "reuse_interp" and self._cache_bus.cons_skip == 1):
                      self._cache_bus.prev_interp = epsilon_m1.clone()

                  if cache_mode == "reuse_interp" and self._cache_bus.cons_skip % 2 == 1:
                      # alternating between trivial psi_t and trivial psi_t-1 interp
                      epsilon_m1 = self._cache_bus.prev_interp

                  pred_m_m_1 = self.convert_model_output(epsilon_m1, sample=prev_sample)
                  self._cache_bus.pred_m_m_1 = pred_m_m_1.clone()

              elif interp_mode == "x_0":
                  # interpolate on trajectory (x_0)
                  pred_prev_sample = sample - 0.625 * delta * f - 0.75 * delta * self._cache_bus.prev_f[-1] + 0.375 * delta * self._cache_bus.prev_f[-2]
                  pred_prev_sample = pred_prev_sample.to(model_output.dtype)
                  pred_m_m_1 = self.convert_model_output(epsilon_0, sample=pred_prev_sample)
                  self._cache_bus.pred_m_m_1 = pred_m_m_1.clone()

              else: raise RuntimeError

          else:
              pred_m_m_1 = lagrange_skip(self._cache_bus.lagrange_step, self._cache_bus.lagrange_x0, self._step_index)
              self._cache_bus.pred_m_m_1 = pred_m_m_1.clone()

      # update on y (f)
      for j in range(1):
          self._cache_bus.prev_f[j] = self._cache_bus.prev_f[j + 1]
      self._cache_bus.prev_f[-1] = f

      if not return_dict:
          return (prev_sample,)

      return SchedulerOutput(prev_sample=prev_sample)