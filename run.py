import subprocess
import sys

DEFAULT_LOGGER = 'rlbot'

if __name__ == '__main__':

    try:
        from rlbot.utils import public_utils, logging_utils

        logger = logging_utils.get_logger(DEFAULT_LOGGER)
        if not public_utils.have_internet():
            logger.log(logging_utils.logging_level,
                       'Skipping upgrade check for now since it looks like you have no internet')
        elif public_utils.is_safe_to_upgrade():
            subprocess.call([sys.executable, "-m", "pip", "install", '-r', 'requirements.txt'])
            subprocess.call([sys.executable, "-m", "pip", "install", 'rlbot', '--upgrade'])

            # https://stackoverflow.com/a/44401013
            rlbots = [module for module in sys.modules if module.startswith('rlbot')]
            for rlbot_module in rlbots:
                sys.modules.pop(rlbot_module)

    except ImportError:
        subprocess.call([sys.executable, "-m", "pip", "install", '-r', 'requirements.txt', '--upgrade', '--upgrade-strategy=eager'])

    try:
        from rlbot import runner
        runner.main()
    except Exception as e:
        print("Encountered exception: ", e)
        print("Press enter to close.")
        input()
