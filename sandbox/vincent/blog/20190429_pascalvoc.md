Il y a certainement des dizaines de façons d'attaquer le sujet de l'amélioration de la segmentation. Je voudrais identifier quel est le meilleur angle d'attaque pour rapidement contribuer à la qualité de l'app. Rien de mieux pour cela que de tester la segmentation en conditions réelles. Dans la mesure où le réseau a été entrainé sur PascalVOC12, autant regarder déjà ce que ce ça donne la dessus.

J'ai donc extrait la segmentation de référence, calculer la segmentation telle qu'elle est calculée dans l'app, et calculer celle obtenue avec l'application du guied filter. Je n'ai traité que 200 images de la base de données sur les 2900 qu'elle compte. L'idée est juste d'avoir un feeling de ce qui se passe. Et parmi ces 200, j'en ai sélectionnées quelques unes disponibles ici :

https://www.dropbox.com/sh/1946cbscenpzewg/AABZigV0Qg7n55Wi8rcENtR9a?dl=0

Plusieurs conclusions.

- Atteindre 100% de IoU n'est pas possible car certaines des annotations sont discutables. Certains objets sont oubliés (mouton dans 2007_000175, personne sur 2007_002378).

- Globalement, je trouve que le réseau n'est pas si mauvais que ça. Il arrive même que notre segmentation fasse mieux que la segmentation annotée manuellement (voir 2007_000999).

- Sans post-processing, il est courant de voir un pattern qui a une structure horizontale et verticale :

- Le guided filter aide pas mal.

En voyant ça, je me dis que la première chose à faire est de nettoyer les erreurs grossières en supprimant par exemple les petites détections parasites ou en supprimant les composantes connexes proches du bord, trop petites, ou minoritaires par rapport à une détection de plus grosse taille, etc.

Il faudrait affiner les contours pour qu'ils collent mieux aux objets (besoin de faire un état de l'art).

Dans l'idéal, il faudrait comparer avec ce que donnerait un réseau plus gros tel que Xception dont Matthieu avait parlé.

Quand je vois la qualité de la segmentation que tu as obtenu Matthieu en utilisant remove.bg sur la souris d'ordinateur, je me dis qu'il y a quelques chose à faire autour de la saillance. Je ne sais pas si remove.bg avait la classe "Computer mouse" dans leurs labels, mais c'est possible.

On pourrait entrainer notre propre réseau sur un dataset plus approprié tel que COCO.

Any thoughts?