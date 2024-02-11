# Enhancing Reinforcement Learning agents with Local Guides 

This is the official implementation of the techniques discussed in the paper [Enhancing Reinforcement Learning agents with Local Guides](https://hal.science/hal-04052358/file/Final_Reinforcement_Learning_with_Local_Guides.pdf).

## Create the conda virtual environment

```
conda create --name rllg python=3.8
conda activate rllg
pip install -e .
```

## Steps to launch it for a new environment

- In the folder `envs`, create a folder with the name of the environment with 3 files:
  - `create_env_name` to create the environment
  - `local_expert_policy` for the local expert
  - `confidence` for the confidence function $\lambda$
- Add the environment in the global files `creation` and `confidence` in `envs`
- Add a config file in `ray_config`
- Modify the `main` file to include the new environment
- Enjoy :)

## Notes regarding the Point-Reach environment

PointReach is based on [Bullet-Safety-Gym](https://github.com/SvenGronauer/Bullet-Safety-Gym), and has been modified internally (directly in their source code) to make it more difficult.

All the details can be found in Appendix B of the paper.

## Visualization

All the results are saved in a ray tune `Experimentanalysis`. You can plot them in the `Visualization.ipynb` notebook.

## License

We follow MIT License. Please see the [License](./LICENSE) file for more information.

**Disclaimer:** This is not an officially supported Huawei Product.


## Credits

This code is built upon the [SimpleSAC Github](https://github.com/young-geng/SimpleSAC), and some wrappers of [gym](https://github.com/openai/gym/tree/master).


## Cite us

If you find this technique useful and you use it in a project, please cite it:
```
@inproceedings{daoudi2023enhancing,
  title={Enhancing Reinforcement Learning Agents with Local Guides},
  author={Daoudi, Paul and Robu, Bogdan and Prieur, Christophe and Dos Santos, Ludovic and Barlier, Merwan},
  booktitle={Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems},
  pages={829--838},
  year={2023}
}
```
