# Simulation

This directory contains the closed-loop multi-agent simulation scaffold used to generate the interaction trajectories analyzed in the manuscript.

The implementation includes round-based interaction, agent action ordering, referee-mediated message handling, routing, short-term conversational context, retrieval-backed memory, and transcript logging.

For the complete interaction protocol and experimental settings, see the manuscript and Supplementary Information.

## Entry point

From the repository root:

```bash
python -m simulation.standard_simulation_runner
```

The runner delegates to the simulation components in this directory.

## Configuration

Run-level settings are defined in:

```text
simulation/config.py
```

This includes model/backend configuration, decoding settings, run length, memory settings, retrieval budget, and output locations.

Adjust the configuration before starting a new simulation.

## Main components

```text
agents.py
environment.py
llm_clients.py
memory.py
referee.py
retrieval.py
runner.py
token_utils.py
```

These modules implement the corresponding components of the simulation scaffold.

## Credentials

External API credentials are supplied through environment variables and are not stored in the repository.

The required credentials depend on the backends selected in the simulation configuration.

## Outputs

The simulation writes the interaction transcript and associated runtime records to the locations defined in `config.py`.

For the experimental conditions and analysis cohorts used in the manuscript, see the manuscript.
