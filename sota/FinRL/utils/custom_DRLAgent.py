from finrl.agents.stablebaselines3.models import DRLAgent, MODELS
from model import CustomDDPG  # Import your custom model class

class CustomDRLAgent(DRLAgent):
    """
    Extends DRLAgent to allow custom model classes.
    Only modifies get_model() to accept both string names and custom classes.
    """
    
    def get_model(self, model_name, model_class=None, model_kwargs=None, policy="MlpPolicy", policy_kwargs=None):
        """
        Get a model - accepts either string name (looks up in MODELS) or custom class.
        
        Args:
            model_name: String name for logging purposes
            model_class: Custom model class (if None, looks up model_name in MODELS)
            model_kwargs: Hyperparameters for the model
            policy: Policy type (default "MlpPolicy")
            policy_kwargs: Policy architecture parameters
        
        Returns:
            Trained model instance
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Determine which class to use
        if model_class is not None:
            # Explicit model_class provided
            target_class = model_class
        elif isinstance(model_name, str):
            # String name - lookup in MODELS
            if model_name not in MODELS:
                raise NotImplementedError(f"Model {model_name} not implemented")
            target_class = MODELS[model_name]
        else:
            # model_name is actually a class (for backward compatibility)
            target_class = model_name
        
        # Add policy_kwargs if provided
        if policy_kwargs:
            model_kwargs["policy_kwargs"] = policy_kwargs
        
        # Create the model
        model = target_class(
            policy=policy,
            env=self.env,
            **model_kwargs
        )
        
        return model